"""
Train the GridNav predictive model, evaluate decode accuracy, and
demonstrate lookahead-planning navigation on top of it.

Run: python3 train_and_test.py [world_file] [n_train_steps]
     python3 train_and_test.py                         # 5x5_maze.world, 9000 steps (~1 min)
     python3 train_and_test.py 9x9_maze.world 25000    # bigger maze (~3 min)

Compares three planners over every start->goal pair:
  1-step greedy  : the original -- imagine 4 actions, pick the one closest to the goal
  multi-step     : imagine all action sequences `depth` steps ahead, same distance score
  SR multi-step  : same imagined rollouts, scored by a learned successor representation
See lookahead.py for details.
"""
import sys
import random
import numpy as np
from popcode import PopCode2D
from world import Env, Action, Pos, move
from predictive import build_predictive_network, run_trial
from lookahead import (PlaceMemory, SuccessorMap, WorldModel, euclidean_value, navigate,
                       shortest_path_len)


def print_reference_points(pc: PopCode2D):
    center = Pos(pc.rows // 2, pc.cols // 2)
    perfect = pc.encode(center)

    def err_at(dr, dc):
        other = pc.encode(Pos(center.row + dr, center.col + dc))
        return float(((perfect - other) ** 2).sum())

    rng = np.random.default_rng(0)
    rand_errs = []
    for _ in range(200):
        target = pc.encode(Pos(rng.integers(1, pc.rows - 1), rng.integers(1, pc.cols - 1)))
        rand_pred = rng.uniform(0, 1, (pc.rows, pc.cols))
        rand_errs.append(float(((target - rand_pred) ** 2).sum()))

    print("Reference points (what a given sq_error number means):")
    print(f"  perfect match       : 0.00")
    print(f"  off by 1 grid cell  : {err_at(0, 1):.2f}")
    print(f"  off by 2 grid cells : {err_at(0, 2):.2f}")
    print(f"  untrained/random    : ~{np.mean(rand_errs):.2f}")
    print()


def train(net, env, pc, n_steps: int, memory: PlaceMemory, sr: SuccessorMap,
          lrate: float = 0.02, report_every: int = 1500):
    """
    Random-walk exploration. Each step trains the predictive network (CHL),
    stores the current place in PlaceMemory, and updates the SR by TD.
    """
    errors = []
    for step in range(n_steps):
        action = random.choice(list(Action))
        before = env.pos
        memory.store(before, pc.encode(before))
        predicted, actual, _, _ = run_trial(net, env, action, pc, lrate=lrate)
        sr.update(before, env.pos)
        errors.append(float(((actual - predicted) ** 2).sum()))
        if (step + 1) % report_every == 0:
            recent = errors[-report_every:]
            print(f"  steps {step - report_every + 2:5d}-{step + 1:5d}: avg sq_error = {sum(recent)/len(recent):.3f}")
    memory.store(env.pos, pc.encode(env.pos))


def evaluate_decode_accuracy(net, env, pc, n_eval: int = 300):
    exact, within_1 = 0, 0
    for _ in range(n_eval):
        action = random.choice(list(Action))
        before = env.pos

        predicted, actual, _, _ = run_trial(net, env, action, pc, lrate=0.0)  # lrate=0 -> no learning
        true_next = move(before, action)
        if not env.world.is_open(true_next):
            true_next = before
        decoded = pc.decode(predicted)
        dist = ((decoded[0] - true_next.row) ** 2 + (decoded[1] - true_next.col) ** 2) ** 0.5
        if dist < 0.5:
            exact += 1
        if dist < 1.5:
            within_1 += 1
    print(f"\nDecode accuracy over {n_eval} held-out trials:")
    print(f"  exact-cell match : {exact}/{n_eval} = {exact/n_eval:.1%}")
    print(f"  within 1 cell    : {within_1}/{n_eval} = {within_1/n_eval:.1%}")


def make_planners(net, pc, memory, sr, depth: int):
    """name -> (fresh WorldModel factory, value_fn, depth)."""
    new_model = lambda: WorldModel(net, pc, memory)
    return {
        "1-step greedy (original)": (new_model, euclidean_value, 1),
        f"{depth}-step lookahead": (new_model, euclidean_value, depth),
        f"SR {depth}-step lookahead": (new_model, sr.value, depth),
    }


def navigate_demo(env, planners, start: Pos, goal: Pos):
    shortest = shortest_path_len(env.world, start, goal)
    max_steps = shortest + 10
    print(f"Start ({start.row},{start.col}) -> goal ({goal.row},{goal.col}), shortest path = {shortest} steps")
    for i, (name, (new_model, value_fn, depth)) in enumerate(planners.items()):
        env.pos = env.prev_pos = start
        last = i == len(planners) - 1  # step-by-step imagined plans for the best planner only
        print(f"\n[{name}]")
        path = navigate(env, new_model(), goal, value_fn, depth=depth, max_steps=max_steps, verbose=last)
        if not last:
            trail = " ".join(f"({p.row},{p.col})" for p in path[:16])
            print(f"path: {trail}{' ...' if len(path) > 16 else ''}")
        if env.pos == goal:
            print(f"REACHED GOAL in {len(path) - 1} steps!")
        else:
            print(f"Did not reach goal within {max_steps} steps. Final position: {env.pos}")


def benchmark_all_pairs(env, planners, cells, max_steps: int = 100):
    """Every start->goal pair. Model corrections persist across episodes, as an agent's would."""
    print(f"\nNavigation over all {len(cells) * (len(cells) - 1)} start->goal pairs:")
    for name, (new_model, value_fn, depth) in planners.items():
        model = new_model()
        reached = optimal = total = 0
        for start in cells:
            for goal in cells:
                if start == goal:
                    continue
                env.pos = env.prev_pos = start
                path = navigate(env, model, goal, value_fn, depth=depth, max_steps=max_steps)
                total += 1
                if env.pos == goal:
                    reached += 1
                    optimal += len(path) - 1 == shortest_path_len(env.world, start, goal)
        print(f"  {name:26s}: reached {reached/total:6.1%}   optimal path {optimal/total:6.1%}"
              f"   ({len(model.corrections)} model corrections)")


if __name__ == "__main__":
    world_file = sys.argv[1] if len(sys.argv) > 1 else "5x5_maze.world"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
    depth = 4

    random.seed(0)
    np.random.seed(0)
    env = Env(world_file)
    pc = PopCode2D(rows=env.world.rows, cols=env.world.cols, sigma=1.0)
    net = build_predictive_network(grid_shape=(env.world.rows, env.world.cols))
    memory = PlaceMemory()
    sr = SuccessorMap(env.world.rows, env.world.cols)

    print_reference_points(pc)

    print(f"Training on {world_file} ({n_train} steps)...")
    train(net, env, pc, n_train, memory, sr)

    evaluate_decode_accuracy(net, env, pc, n_eval=300)

    cells = [Pos(r, c) for r in range(env.world.rows) for c in range(env.world.cols)
             if env.world.grid[r][c]]
    print(f"\nPlace memory holds {len(memory)}/{len(cells)} open cells.")
    planners = make_planners(net, pc, memory, sr, depth)

    # Demo: from the top-left corner to the open cell farthest away along the maze.
    start = Pos(1, 1)
    goal = max(cells, key=lambda g: shortest_path_len(env.world, start, g))
    print("\n--- Lookahead-planning navigation demo ---")
    navigate_demo(env, planners, start, goal)

    benchmark_all_pairs(env, planners, cells)

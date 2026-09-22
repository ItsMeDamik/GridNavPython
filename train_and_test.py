"""
Train the GridNav predictive model, evaluate decode accuracy, and
demonstrate lookahead-planning navigation on top of it.

Run: python3 train_and_test.py
Expected result: ~98% exact-cell decode accuracy, and the navigation
demo reaching its goal in the optimal number of steps.
"""
import random
import numpy as np
from popcode import PopCode2D
from world import Env, Action, Pos, move
from predictive import build_predictive_network, run_trial, imagine


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


def train(net, env, pc, n_steps: int, lrate: float = 0.02, report_every: int = 1500):
    errors = []
    for step in range(n_steps):
        action = random.choice(list(Action))
        predicted, actual, _, _ = run_trial(net, env, action, pc, lrate=lrate)
        errors.append(float(((actual - predicted) ** 2).sum()))
        if (step + 1) % report_every == 0:
            recent = errors[-report_every:]
            print(f"  steps {step - report_every + 2:5d}-{step + 1:5d}: avg sq_error = {sum(recent)/len(recent):.3f}")


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


def choose_action_by_lookahead(net, pos_pattern, goal: Pos, pc: PopCode2D):
    """Pure lookahead planning: no learned policy. Imagine all 4 actions,
    pick whichever predicted outcome is closest to the goal."""
    best_a, best_dist = None, float("inf")
    for a in range(4):
        pred = imagine(net, pos_pattern, a, pc)
        decoded = pc.decode(pred)
        dist = ((decoded[0] - goal.row) ** 2 + (decoded[1] - goal.col) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best_a = dist, a
    return best_a, best_dist


def navigate_demo(net, env, pc, start: Pos, goal: Pos, max_steps: int = 15):
    env.pos = start
    print(f"\nStart: {env.pos}, Goal: {goal}\n")
    for step in range(max_steps):
        if env.pos == goal:
            print(f"REACHED GOAL in {step} steps!")
            return
        pos_pattern = pc.encode(env.pos)
        action_idx, pred_dist = choose_action_by_lookahead(net, pos_pattern, goal, pc)
        action = Action(action_idx)
        before = env.pos
        env.take_action(action)
        print(f"step {step:2d}: at {before} -> chose {action.name:6s} (pred_dist={pred_dist:.2f}) -> now at {env.pos}")
    print(f"Did not reach goal within {max_steps} steps. Final position: {env.pos}")


if __name__ == "__main__":
    random.seed(0)
    env = Env("5x5_maze.world")
    pc = PopCode2D(rows=env.world.rows, cols=env.world.cols, sigma=1.0)
    net = build_predictive_network(grid_shape=(env.world.rows, env.world.cols))

    print_reference_points(pc)

    print("Training (9000 steps)...")
    train(net, env, pc, n_steps=9000)

    evaluate_decode_accuracy(net, env, pc, n_eval=300)

    print("\n--- Lookahead-planning navigation demo ---")
    navigate_demo(net, env, pc, start=Pos(1, 1), goal=Pos(5, 5))

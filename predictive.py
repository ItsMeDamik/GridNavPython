"""
Predictive network.

Adds a real InputP layer (the "Pulvinar" / predictive
counterpart of Input) and wires Hidden -> InputP as the core 
predictive pathway.

Each trial runs in two phases:

Minus phase: current position + chosen action are clamped in.
Hidden computes from those (Input+Action -> Hidden).
Then Hidden -> InputP generates a Prediction of the next position 
-- with no knowledge yet of what actually happens.
Hidden and InputP are allowed to recirculate for several cycles to settle the prediction.

Plus phase: the action is actually taken in the world, producing the 
real next position. That becomes ground truth. InputP is clamped to that, and Hidden recomputes from Input+Action+InputP.

"""

import numpy as np
from network import Network
from popcode import PopCode2D
from world import Env, Action
from fffb import fffb_cycle

N_QUARTERS = 4
CYC_PER_QTR = 25


def build_predictive_network(grid_shape: tuple[int, int], hidden_shape=(15, 15), \
                             seed: int = 0) -> Network:
    net = Network()
    net.add_layer("Input", shape=grid_shape)
    net.add_layer("Action", shape=(4,))
    net.add_layer("Hidden", shape=hidden_shape)
    net.add_layer("InputP", shape=grid_shape)

    net.connect("Input", "Hidden", wt_scale=10.0, seed=seed + 1)
    net.connect("Action", "Hidden", wt_scale=1.0, seed=seed + 2)
    hidden_to_inputp =net.connect("Hidden", "InputP", wt_scale=3.0, seed=seed + 3) # forward: predict next position
    net.connect("InputP", "Hidden", wt_scale=1.0, tied_to=hidden_to_inputp) # backward: recirculate the outcome

    return net


def run_trial(net: Network, env: Env, action: Action, pc: PopCode2D, lrate: float = 0.02, n_outer: int = 5):
    """
    Run one minus/plus trial. Returns (predicted, actial, squared_error, moved).
    """
    Input, Action, Hidden, InputP = net.layers["Input"], net.layers["Action"], net.layers["Hidden"], net.layers["InputP"]

    for l in (Hidden, InputP):
        l.fbi = 0.0
        l.act = np.zeros(l.shape)
        l.clamped = False

    Input.clamp(pc.encode(env.pos))
    action_pattern = np.zeros(4)
    action_pattern[action.value] = 1.0
    Action.clamp(action_pattern)

    # compute the real outcome NOW (env is deterministic) -- needed for the
    # plus-phase clamp, but don't execute the move until after settling
    from world import move
    next_pos = move(env.pos, action)
    if not env.world.is_open(next_pos):
        next_pos = env.pos
    actual_pattern = pc.encode(next_pos)

    predicted, hidden_minus = None, None

    for cyc in range(N_QUARTERS * CYC_PER_QTR):
        if cyc == 3 * CYC_PER_QTR:                          # minus phase ends, plus phase begins
            predicted = InputP.act.copy()       # capture the free-running guess HERE
            hidden_minus = Hidden.act.copy()
            InputP.clamp(actual_pattern)         # mid-loop clamp, cycling continues uninterrupted
        net.cycle()

    hidden_plus = Hidden.act.copy()

    moved = env.take_action(action)
    actual = pc.encode(env.pos)                # should equal actual_pattern exactly
    sq_error = float(((actual - predicted) ** 2).sum())


    # --- Weight update ---
    for proj in net.projections:
        if proj.send_layer.name == "Input" and proj.recv_layer.name == "Hidden":
            proj.learn(Input.act, hidden_minus, Input.act, hidden_plus, lrate)
        elif proj.send_layer.name == "Action" and proj.recv_layer.name == "Hidden":
            proj.learn(Action.act, hidden_minus, Action.act, hidden_plus, lrate)
        elif proj.send_layer.name == "Hidden" and proj.recv_layer.name == "InputP":
            proj.learn(hidden_minus, predicted, hidden_plus, actual, lrate)
        elif proj.send_layer.name == "InputP" and proj.recv_layer.name == "Hidden":
            pass # tied to Hidden->InputP; no independent weights, nothing to learn
        else:
            raise ValueError(f"Unexpected projection: {proj.send_layer.name} -> {proj.recv_layer.name}")

    InputP.unclamp()

    return predicted, actual, sq_error, moved



def imagine(net: Network, pos_pattern: np.ndarray, action: Action, pc: PopCode2D, n_cycles: int = 75) -> np.ndarray:
    """Pure imagination: free minus-phase settle, no environment, no
    learning. Used for lookahead planning -- 'what would happen if...'"""
    Input, Actn, Hidden, InputP = (net.layers[n] for n in ("Input", "Action", "Hidden", "InputP"))
    for l in (Hidden, InputP):
        l.fbi = 0.0
        l.act = np.zeros(l.shape)
        l.clamped = False
    Input.act = pos_pattern
    Input.clamped = True
    action_pattern = np.zeros(4)
    action_pattern[int(action)] = 1.0
    Actn.act = action_pattern
    Actn.clamped = True
    for _ in range(n_cycles):
        net.cycle()
    return InputP.act.copy()


if __name__ == "__main__":
    import random
    from policy import choose_action

    random.seed(0)
    env = Env("5x5.world")
    pc = PopCode2D(rows=env.world.rows, cols=env.world.cols, sigma=1.0)
    net = build_predictive_network(grid_shape=(env.world.rows, env.world.cols))

    print(f"World: {env.world.rows} x {env.world.cols}, starting at {env.pos}\n")

    for step in range(5):
        action = choose_action(env)
        predicted, actual, sq_err, moved, hidden_minus, hidden_plus = run_trial(net, env, action, pc)
        print(f"step {step}: action={action.name:6s} moved={moved!s:5} "
              f"pos_after={env.pos} prediction_squared_error={sq_err:.3f}")
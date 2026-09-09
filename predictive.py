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


def build_predictive_network(grid_shape: tuple[int, int], hidden_shape=(5, 5), \
                             seed: int = 0) -> Network:
    net = Network()
    net.add_layer("Input", shape=grid_shape)
    net.add_layer("Action", shape=(4,))
    net.add_layer("Hidden", shape=hidden_shape)
    # Note: InputP must be added after Hidden - Network.cycle() processes 
    # non-clamped layers in insertion order, and InputP's prediction depends on
    # Hidden already being computed this same cycle.
    net.add_layer("InputP", shape=grid_shape)

    net.connect("Input", "Hidden", wt_scale=4.0, seed=seed + 1)
    net.connect("Action", "Hidden", wt_scale=1.0, seed=seed + 2)
    net.connect("Hidden", "InputP", wt_scale=4.0, seed=seed + 3) # forward: predict next position
    net.connect("InputP", "Hidden", wt_scale=1.0, seed=seed + 4) # backward: recirculate the outcome

    return net


def run_trial(net: Network, env: Env, action: Action, pc: PopCode2D, lrate: float = 0.02, n_outer: int = 5):
    """
    Run one minus/plus trial. Returns (predicted, actial, squared_error, moved).
    """
    Input, Actn, Hidden, InputP = net.layers["Input"], net.layers["Action"], net.layers["Hidden"], net.layers["InputP"]

    # --- Minus phase ---
    net.layers["Input"].clamp(pc.encode(env.pos))
    action_pattern = np.zeros(4)
    action_pattern[int(action)] = 1.0
    net.layers["Action"].clamp(action_pattern)

    InputP.act = np.zeros(InputP.shape) # don't leak last trial's plus-phase value into this trial

    for _ in range(n_outer):  # Hidden <-> InputP recirculation, to settle the prediction
        net.cycle()     # Input+Action -> Hidden <-> InputP, in one pass

    predicted = net.layers["InputP"].act.copy() # minus-phase guess
    hidden_minus = Hidden.act.copy() # minus-phase hidden

    # --- Plus phase ---
    moved = env.take_action(action)
    actual = pc.encode(env.pos)

    error = actual - predicted
    sq_error = float((error ** 2).sum())

    InputP.clamp(actual) # plus-phase ground truth. InputP is fixed now -- only Hidden is left to compute
    net.cycle() # One settle for Hidden; no other free layer to alternate with
    hidden_plus = Hidden.act.copy() # plus-phase hidden

    # --- Weight update ---
    for proj in net.projections:
        if proj.send_layer.name == "Input" and proj.recv_layer.name == "Hidden":
            proj.learn(Input.act, hidden_minus, Input.act, hidden_plus, lrate)
        elif proj.send_layer.name == "Action" and proj.recv_layer.name == "Hidden":
            proj.learn(Actn.act, hidden_minus, Actn.act, hidden_plus, lrate)
        elif proj.send_layer.name == "Hidden" and proj.recv_layer.name == "InputP":
            proj.learn(hidden_minus, predicted, hidden_plus, actual, lrate)
        elif proj.send_layer.name == "InputP" and proj.recv_layer.name == "Hidden":
            proj.learn(predicted, hidden_minus, actual, hidden_plus, lrate) # think about it
        else:
            raise ValueError(f"Unexpected projection: {proj.send_layer.name} -> {proj.recv_layer.name}")

    InputP.unclamp()

    return predicted, actual, sq_error, moved


if __name__ == "__main__":
    import random
    from policy import choose_action

    random.seed(0)
    env = Env("5x5.world")
    pc = PopCode2D(rows=env.world.rows, cols=env.world.cols, sigma=1.0)
    net = build_predictive_network(grid_shape=(env.world.rows, env.world.cols))

    print(f"World: {env.world.rows} x {env.world.cols}, starting at {env.pos}\n")

    for step in range(30):
        action = choose_action(env)
        predicted, actual, sq_err, moved = run_trial(net, env, action, pc)
        print(f"step {step}: action={action.name:6s} moved={moved!s:5} "
              f"pos_after={env.pos} prediction_squared_error={sq_err:.3f}")
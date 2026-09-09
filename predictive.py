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

Plus phase: the action is actually taken in the world, producing the 
real next position. That becomes ground truth.

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
    net.connect("Hidden", "InputP", wt_scale=4.0, seed=seed + 3)
    return net


def run_trial(net: Network, env: Env, action: Action, pc: PopCode2D):
    """
    Run one minus/plus trial. Returns (predicted, actial, squared_error, moved).
    """

    # --- Minus phase ---
    net.layers["Input"].clamp(pc.encode(env.pos))
    action_pattern = np.zeros(4)
    action_pattern[int(action)] = 1.0
    net.layers["Action"].clamp(action_pattern)

    net.cycle()     # Input+Action -> Hidden -> InputP, in one pass

    predicted = net.layers["InputP"].act.copy()

    # --- Plus phase ---
    moved = env.take_action(action)
    actual = pc.encode(env.pos)

    error = actual - predicted
    sq_error = float((error ** 2).sum())

    return predicted, actual, sq_error, moved


if __name__ == "__main__":
    import random
    from policy import choose_action

    random.seed(0)
    env = Env("5x5.world")
    pc = PopCode2D(rows=env.world.rows, cols=env.world.cols, sigma=1.0)
    net = build_predictive_network(grid_shape=(env.world.rows, env.world.cols))

    print(f"World: {env.world.rows} x {env.world.cols}, starting at {env.pos}\n")

    for step in range(6):
        action = choose_action(env)
        predicted, actual, sq_err, moved = run_trial(net, env, action, pc)
        print(f"step {step}: action={action.name:6s} moved={moved!s:5} "
              f"pos_after={env.pos} prediction_squared_error={sq_err:.3f}")
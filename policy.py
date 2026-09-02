"""
Policy for grid navigation.

A simple "subcortical" behavior policy: pick a random action, but reject it if 
it would walk into a wall, retrying a limited number of times. This generates 
the (position, action, next_position) experience that the network will later 
learn to predict.
"""

import random
from world import Action, Env, move


def choose_action(env: Env, max_tries: int = 10) -> Action:
    """
    Pick a random action that won't walk into a wall.
    Falls back to a plain random action (even into a wall) if every attempt
    failed -- e.g. agent is boxed in on 3 sides + tries ran out. 
    """
    for _ in range(max_tries):
        candidate = random.choice(list(Action))
        target = move(env.pos, candidate)
        if env.world.is_open(target):
            return candidate
    # exhausted all tries -- just return something, Env.take_action() will
    # simple reject it and leave the agent in place.
    return random.choice(list(Action))


if __name__ == "__main__":
    random.seed(0)
    env = Env("5x5.world")

    trajectory = [env.pos]
    wall_bumps = 0

    for step in range(20):
        action = choose_action(env)
        moved = env.take_action(action)
        if not moved:
            wall_bumps += 1
        trajectory.append(env.pos)
        print(f"step {step:2d}: action={action.name:6s} -> {env.pos} \
              {'' if moved else '(blocked!)'}")

        print(f"\nWall bumps out of 20 steps: {wall_bumps}")
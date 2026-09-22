import random
import numpy as np
from predictive import build_predictive_network, run_trial
from world import Env, Action, move
from popcode import PopCode2D

random.seed(0)

env = Env('5x5.world')
pc = PopCode2D(rows=env.world.rows, cols=env.world.cols, sigma=1.0)
net = build_predictive_network(grid_shape=(env.world.rows, env.world.cols), seed=0)
open_cells = [(r, c) for r in range(env.world.rows) for c in range(env.world.cols) if env.world.grid[r][c]]

n_trials = 5000
correct = 0
invalid = 0

for _ in range(n_trials):
    row, col = random.choice(open_cells)
    env.pos = env.prev_pos = type(env.pos)(row, col)
    action = random.choice(list(Action))
    true_next = move(env.pos, action)
    if not env.world.is_open(true_next):
        true_next = env.pos

    predicted, actual, sq_err, moved = run_trial(net, env, action, pc, lrate=0.02)
    decoded = pc.decode(predicted)
    if not np.all(np.isfinite(decoded)):
        invalid += 1
        exact = False
    else:
        row_dec = int(round(decoded[0]))
        col_dec = int(round(decoded[1]))
        exact = (row_dec == true_next.row) and (col_dec == true_next.col)
        correct += int(exact)
        

acc = 100.0 * correct / n_trials
print(f'Exact-cell decode accuracy over {n_trials} trials: {acc:.2f}% ({correct}/{n_trials})')
print(f'Invalid/NaN predictions: {invalid}/{n_trials} ({100.0 * invalid / n_trials:.2f}%)')

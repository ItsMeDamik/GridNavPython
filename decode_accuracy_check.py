import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
trial_results = []

for trial_idx in range(1, n_trials + 1):
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

    trial_results.append(int(exact))

    if trial_idx % 25 == 0 or trial_idx == 1:
        cum_acc = 100.0 * sum(trial_results) / trial_idx
        # optional: keep a lightweight running trace for drift diagnostics
        pass

cumulative_acc = np.cumsum(trial_results) / np.arange(1, n_trials + 1)
trial_axis = np.arange(1, n_trials + 1)

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(trial_axis, 100.0 * cumulative_acc, color='tab:blue', linewidth=2)
ax.set_title('Exact-cell decode accuracy vs trials')
ax.set_xlabel('Trial')
ax.set_ylabel('Cumulative accuracy (%)')
ax.grid(True, alpha=0.3)

fig.suptitle(f'Exact-cell decode accuracy over {n_trials} trials')
fig.savefig('decode_accuracy_vs_trials.png', dpi=200)

acc = 100.0 * correct / n_trials
print(f'Exact-cell decode accuracy over {n_trials} trials: {acc:.2f}% ({correct}/{n_trials})')
print(f'Invalid/NaN predictions: {invalid}/{n_trials} ({100.0 * invalid / n_trials:.2f}%)')
print('Saved plot to decode_accuracy_vs_trials.png')


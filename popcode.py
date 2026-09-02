"""
Population Coding.

Represents a continuous/discrete (row, col) position as a distributed pattern
of activity over a (rows x cols) grid of "place cells" -- each unit fires 
maximally when the agent is at the cell's own location, falling off as 
a Gaussian with distance. 

Note that unit is the place cell.
"""

import numpy as np
from world import Pos


class PopCode2D:
    """
    Encodes a 2D position as a Gaussian bump over a grid of units, and decodes
    a pattern of activity back into an estimated position
    """

    def __init__(self, rows: int, cols: int, sigma: float = 1.0):
        self.rows = rows
        self.cols = cols
        self.sigma = sigma
        # Precompute each unit's own preferred (row, col) location once --
        # unit (r, c) in the pattern IS the place cell centered at (r, c).
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        self.unit_row = rr # shape (rows, cols)
        self.unit_col = cc

    def encode(self, pos: Pos) -> np.ndarray:
        """
        Returns a (rows, cols) array: activity of every unit given the agent's 
        actual position.
        """
        d2 = (self.unit_row - pos.row) ** 2 + (self.unit_col - pos.col) ** 2 # squared Euclidean distance to every unit simultaneously
        act = np.exp(-d2 / (2 * self.sigma ** 2)) # standard Gaussian formula exp(-distance^2 / 2sigma^2)
        return act

    def decode(self, pattern: np.ndarray) -> tuple[float, float]:
        """
        Population-based readout: the activity-weighted average of all units'
        preferred locations. Given a clean encode() output this should recover 
        the original position (approximately).
        """
        total = pattern.sum()
        if total <= 1e-8:
            return (float("nan"), float("nan"))
        # weighted average: for every single unit, multiply its activation by its 
        # own row-address — then add all those products together — then divide 
        # by total activation
        row_est = (pattern * self.unit_row).sum() / total
        col_est = (pattern * self.unit_col).sum() / total
        return (row_est, col_est)



def ascii_heatmap(pattern: np.ndarray) -> str:
    """
    Cheap visualization: map activity 0...1 to characters, no plotting lib needed.
    """
    chars = " .:-=+*#%@"
    lines = []
    for row in pattern:
        line = "".join(chars[min(int(v * (len(chars) - 1)), len(chars) - 1)] for v in row)
        lines.append(line)
    return "\n".join(lines)

if __name__ == "__main__":
    pc = PopCode2D(rows=7, cols=7, sigma=1.0)

    pos = Pos(row=1, col=1)
    pattern = pc.encode(pos)
    print(pc.unit_row)
    print(pc.unit_col)

    print(f"Encoding position {pos}: \n")
    print(ascii_heatmap(pattern))

    decoded = pc.decode(pattern)
    print(f"\nDecoded back to: (row={decoded[0]:.2f}, col={decoded[1]:.2f})")
    print(f"Original was:   (row={pos.row}, col={pos.col})")
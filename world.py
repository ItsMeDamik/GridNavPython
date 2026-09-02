"""
This module defines a simple grid world environment.
'#' = wall, ' ' = open space. Agent lives at integer (row, col).
""" 

from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

ACTION_DELTA = {
    Action.NORTH: (-1, 0),
    Action.EAST: (0, 1),
    Action.SOUTH: (1, 0),
    Action.WEST: (0, -1),
}

@dataclass(frozen=True)
class Pos:
    row: int
    col: int

class World:
    def __init__(self, filename: str):
        with open(filename, 'r') as f:
            lines = [line.rstrip("\n") for line in f if line.strip("\n") != ""]
        self.rows = len(lines)
        self.cols = len(lines[0])
        self.grid = [[ch == " " for ch in line] for line in lines]

    def is_open(self, pos: Pos) -> bool:
        if not (0 <= pos.row < self.rows and 0 <= pos.col < self.cols):
            return False
        return self.grid[pos.row][pos.col]

def move(pos: Pos, action: Action) -> Pos:
    dr, dc = ACTION_DELTA[action]
    return Pos(pos.row + dr, pos.col + dc)


class Env:
    def __init__(self, world_file: str, start: Pos = None):
        self.world = World(world_file)
        self.pos = start or Pos(1, 1)
        self.prev_pos = self.pos

    def take_action(self, action: Action) -> bool:
        new_pos = move(self.pos, action)
        if self.world.is_open(new_pos):
            self.prev_pos = self.pos
            self.pos = new_pos
            return True
        return False

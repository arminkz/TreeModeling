from __future__ import annotations
from sca_bud.attractor import Attractor
import numpy as np


class Node:
    influenced_by: [Attractor]
    is_fresh: bool
    parent: Node
    children: [Node]
    heading: np.ndarray
    thickness: float
    is_tip: bool
    is_bud: bool
    bud_heading: np.ndarray
    last_bud_heading: np.ndarray
    dist_to_root: int
    dist_to_bud: int
    branch_depth: int

    def __init__(self, position, heading, parent):
        self.position = np.array(position,dtype='float64')
        self.heading = np.array(heading,dtype='float64')
        self.parent = parent
        self.children = []
        self.is_fresh = True
        self.influenced_by = []
        self.thickness = 0
        self.is_tip = True
        self.is_bud = False
        self.bud_heading = None

        if parent is not None:
            self.dist_to_root = parent.dist_to_root + 1
            self.dist_to_bud = parent.dist_to_bud + 1
            self.branch_depth = parent.branch_depth
            self.last_bud_heading = parent.last_bud_heading
        else:
            self.dist_to_root = 0
            self.dist_to_bud = 0
            self.branch_depth = 0
            self.last_bud_heading = None

    def __str__(self):
        return f"Node at {self.position}"

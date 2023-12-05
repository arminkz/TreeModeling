from __future__ import annotations

import numpy as np


class Node:
    is_fresh: bool
    position: np.ndarray
    parent: Node
    children: [Node]
    thickness: float
    is_merging: bool

    def __init__(self, position, children):
        self.position = position
        self.children = children
        self.is_fresh = True
        # Parent and thickness are not determined at construction time
        self.parent = None
        self.thickness = 0
        self.is_merging = False

    def __str__(self):
        return f"Node at {self.position}"

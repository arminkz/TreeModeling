from __future__ import annotations
from .attractor import Attractor


class Node:
    influenced_by: [Attractor]
    is_fresh: bool
    parent: Node
    children: [Node]
    thickness: float
    is_degen: bool

    def __init__(self, position, parent):
        self.position = position
        self.parent = parent
        self.children = []
        self.is_fresh = True
        self.influenced_by = []
        self.thickness = 0
        self.is_degen = False

    def __str__(self):
        return f"Node at {self.position}"

from attractor import Attractor


class Node:
    influenced_by: [Attractor]
    is_fresh: bool

    def __init__(self, position, parent):
        self.position = position
        self.parent = parent
        self.is_fresh = True
        self.influenced_by = []

    def __str__(self):
        return f"Node at {self.position}"

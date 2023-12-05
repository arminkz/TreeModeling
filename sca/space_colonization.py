import numpy as np

from scipy.spatial import KDTree
from .attractor import Attractor
from .node import Node


class SpaceColonization:
    attraction_points: [Attractor]
    nodes: [Node]
    root_node: Node
    kdtree: KDTree

    # constant parameters
    attraction_dist: float = 0.6
    kill_distance: float = 0.05
    step_size: float = 0.02

    def __init__(self, attraction_points, root_pos):
        # Create a set of attraction points
        self.attraction_points = attraction_points

        # Create a set of nodes
        self.nodes = []
        self.root_node = Node(root_pos, parent=None)
        self.nodes.append(self.root_node)

        # Build spatial Index
        self.kdtree = None
        self.build_spatial_index()

    def build_spatial_index(self):
        # Create KD-Tree for nodes
        nd = np.array([node.position for node in self.nodes])
        self.kdtree = KDTree(nd)

    def check_kill_distance(self, attractor):
        return len(self.kdtree.query_ball_point(attractor.position, r=self.kill_distance)) > 0

    def get_closest_node(self, attractor):
        return self.kdtree.query(attractor.position, distance_upper_bound=self.attraction_dist)

    def update(self):
        # Clear all influence data
        for n in self.nodes:
            n.influenced_by = []
        for a in self.attraction_points:
            a.influencing_nodes = []

        # Calculate influence
        for a in self.attraction_points:
            d, ni = self.get_closest_node(a)
            if d != np.inf:
                n = self.nodes[ni]
                n.influenced_by.append(a)
                a.influencing_nodes.append(n)

        # Create new nodes
        fresh_nodes = []
        for n in self.nodes:
            if len(n.influenced_by) > 0:
                direction = np.array([.0, .0, .0])
                atr: Attractor
                for atr in n.influenced_by:
                    direction += atr.position - n.position

                # Special degen case
                if len(n.influenced_by) == 2:
                    center = (n.influenced_by[0].position + n.influenced_by[1].position) / 2
                    if np.linalg.norm(n.position - center) < self.step_size:
                        direction = n.influenced_by[0].position - n.position

                direction /= np.linalg.norm(direction)
                new_pos = n.position + direction * self.step_size
                new_node = Node(new_pos, parent=n)
                # Set as child in n
                n.children.append(new_node)
                fresh_nodes.append(new_node)

        for n in fresh_nodes:
            self.nodes.append(n)
        # print(f"{len(fresh_nodes)} nodes added.")

        killables = []
        for a in self.attraction_points:
            if self.check_kill_distance(a):
                killables.append(a)
        for ka in killables:
            self.attraction_points.remove(ka)

        # Refresh KD-Tree
        self.build_spatial_index()
        # Return if the algorithm has made changes
        return len(fresh_nodes) > 0 or len(killables) > 0

    def run(self):
        while self.update():
            pass

    def set_thickness_values(self):
        # Recursive DFS traversal
        def set_thickness_values(node):
            thickness = 0.0
            if len(node.children) == 0:
                thickness = 1.0
            elif len(node.children) == 1:
                thickness = set_thickness_values(node.children[0])
            else:
                thickness = 0.5
                for child in node.children:
                    thickness += 0.5 * set_thickness_values(child)
            node.thickness = thickness
            return thickness

        # Set thickness for all nodes
        set_thickness_values(self.root_node)

    def get_skeleton(self):
        self.run()
        self.set_thickness_values()

        all_segments = []

        def segment_traverse(node):
            # Leaf
            if len(node.children) == 0:
                return [(node.thickness, node.position)]
            # Segment
            if len(node.children) == 1:
                return [(node.thickness, node.position)] + segment_traverse(node.children[0])
            # Branch
            if len(node.children) >= 2:
                for cn in node.children:
                    new_segment(cn, parent_element=[(node.thickness, node.position)])
                return [(node.thickness, node.position)]

        def new_segment(start_node, parent_element):
            if parent_element is None:
                all_segments.append(segment_traverse(start_node))
            else:
                all_segments.append( parent_element + segment_traverse(start_node) )

        new_segment(self.root_node, parent_element=None)

        return all_segments

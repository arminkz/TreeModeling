import numpy as np
import open3d as o3d
import random
import math
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from pf.node import Node


class ParticleFlow:
    nodes: [Node]
    fresh_nodes: [Node]
    root_pos: np.ndarray
    root_node: Node
    kdtree: KDTree

    merge_distance = 0.04
    step_size: float = 0.02

    def __init__(self, initial_points, root_pos):
        # Create set of initial nodes (leaf nodes)
        self.nodes = []
        for pos in initial_points:
            self.nodes.append(Node(pos, []))

        # Root position
        self.root_pos = root_pos
        self.root_node = Node(root_pos,[])

        # Build spatial Index
        self.kdtree = None
        self.build_spatial_index()

    def build_spatial_index(self):
        # Create KD-Tree for nodes
        self.fresh_nodes = []
        for n in self.nodes:
            if n.is_fresh:
                self.fresh_nodes.append(n)
        if len(self.fresh_nodes) == 0:
            print("no fresh nodes!")
            return
        nd = np.array([n.position for n in self.fresh_nodes])
        self.kdtree = KDTree(nd)

    def get_closest_fresh_node(self, origin_node):
        ds , nis = self.kdtree.query(origin_node.position, k=2)
        distance = ds[1]
        if distance != np.inf:
            return distance, self.fresh_nodes[nis[1]]
        else:
            print("no nearest neighbour")
            return distance, None

    def draw_point(self, pt, color):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        s.paint_uniform_color(color)
        s.compute_vertex_normals()
        s.translate(pt)
        return s

    def visualize_with_spheres(self):
        # Define some colors
        color_red = [1, 0, 0]
        color_blue = [0, 0, 1]

        # Drawing code
        all_drawables = []

        # Draw nodes
        for n in self.nodes:
            if n.is_fresh:
                all_drawables.append(self.draw_point(n.position, color_blue))
            else:
                all_drawables.append(self.draw_point(n.position, color_red))
        return all_drawables

    def update(self):

        # check merging of two nodes
        all_pos = []
        all_fresh_nodes = []
        for node in self.nodes:
            if node.is_fresh:
                all_fresh_nodes.append(node)
                all_pos.append(node.position)

        if len(all_fresh_nodes) == 0:
            return False

        dbscan = DBSCAN(eps=self.merge_distance, min_samples=1).fit(np.array(all_pos))
        lbls = dbscan.labels_

        for l in range(np.max(lbls)):
            idxs = np.asarray(lbls == l).nonzero()[0]
            if len(idxs) > 1:
                print(f"should be merged : {idxs}")
                all_children = []
                sum_pos = np.zeros(3)
                for idx in idxs:
                    all_children += all_fresh_nodes[idx].children
                    sum_pos += all_fresh_nodes[idx].position
                print(f"new children : {all_children}")
                new_pos = sum_pos / len(idxs)
                new_node = Node(new_pos, children=all_children)
                new_node.is_fresh = True
                new_node.is_merging = True
                # add newly constructed node
                self.nodes.append(new_node)
                # remove old ones
                for idx in idxs:
                    self.nodes.remove(all_fresh_nodes[idx])

        # After merging, nodes are invalidated, recalculate spatial index
        self.build_spatial_index()

        # insert new nodes
        added_nodes = []
        for n in self.nodes:
            if n.is_fresh:

                # check if node has reached its target
                if np.linalg.norm(n.position - self.root_pos) < self.merge_distance:
                    n.is_fresh = False
                    if not (n in self.root_node.children):
                        self.root_node.children.append(n)
                    continue

                d, nn = self.get_closest_fresh_node(n)
                if d != np.inf:
                    to_root = self.root_pos - n.position
                    to_root /= np.linalg.norm(to_root)
                    to_nn = nn.position - n.position
                    to_nn /= np.linalg.norm(to_nn)

                    direction = to_root + to_nn
                    direction /= np.linalg.norm(direction)

                    new_pos = n.position + direction * self.step_size
                    new_node = Node(new_pos, [n])
                    new_node.is_fresh = True
                    added_nodes.append(new_node)
                else:
                    print("No nearest neighbour !")
                n.is_fresh = False

        for n in added_nodes:
            self.nodes.append(n)

        # Refresh KD-Tree
        self.build_spatial_index()

        return len(added_nodes) > 0

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
                all_segments.append(parent_element + segment_traverse(start_node))

        new_segment(self.root_node, parent_element=None)

        return all_segments


if __name__ == "__main__":

    def get_random_point_on_sphere():
        x1 = random.random() - 0.5
        x2 = random.random() - 0.5
        x3 = random.random() - 0.5

        vec = np.array([x1, x2, x3])
        vec /= np.linalg.norm(vec)
        return vec + np.array([0.5, 0.5, 0.5])


    def get_random_point_in_sphere():
        u = random.random()
        x1 = random.random() - 0.5
        x2 = random.random() - 0.5
        x3 = random.random() - 0.5

        mag = math.sqrt(x1 * x1 + x2 * x2 + x3 * x3)
        x1 /= mag
        x2 /= mag
        x3 /= mag

        c = u ** (1. / 3.)
        return np.array([x1 * c, x2 * c, x3 * c]) + np.array([0.5, 0.5, 0.5])


    initial_positions = []
    root_pos = np.array([0.5, 0.5, -1])

    n_outer_pts = 100
    for i in range(n_outer_pts):
        rpos = get_random_point_on_sphere()
        initial_positions.append(rpos)
    n_inner_pts = 40
    for i in range(n_inner_pts):
        rpos = get_random_point_in_sphere()
        initial_positions.append(rpos)

    pf = ParticleFlow(initial_positions, root_pos)

    def advance_key_callback(vis: o3d.visualization.Visualizer):
        vis.clear_geometries()
        pf.update()
        for geom in pf.visualize_with_spheres():
            vis.add_geometry(geom, reset_bounding_box=False)

    def run_key_callback(vis: o3d.visualization.Visualizer):
        vis.clear_geometries()
        pf.run()
        for geom in pf.visualize_with_spheres():
            vis.add_geometry(geom, reset_bounding_box=False)

    key_to_callback = {ord("A"): advance_key_callback, ord("D"): run_key_callback}
    o3d.visualization.draw_geometries_with_key_callbacks(pf.visualize_with_spheres(), key_to_callback)
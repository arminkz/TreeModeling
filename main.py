import numpy as np
import open3d as o3d
import random
import math

from scipy.spatial import KDTree
from sca.attractor import Attractor
from sca.node import Node


def pnt2line(pnt, start, end):
    line_vec = end - start
    pnt_vec = pnt - start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    pnt_vec_scaled = pnt_vec * (1.0/line_len)
    t = np.dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_vec * t
    dist = np.linalg.norm(pnt_vec - nearest) # distance(nearest, pnt_vec)
    nearest = nearest + start
    return dist


class SpaceColonization:
    attraction_points: [Attractor]
    nodes: [Node]
    root_node: Node
    kdtree: KDTree

    # constant parameters
    attraction_dist: float = 0.6
    kill_distance: float = 0.05
    step_size: float = 0.02

    def __init__(self):

        # Create a set of attraction points
        self.attraction_points = []
        n_attraction_pts = 600
        for i in range(n_attraction_pts):
            rpos = get_random_point_in_sphere()
            self.attraction_points.append(Attractor(rpos))
        self.attraction_points.append(Attractor(np.array([0.5, 0.5, -0.4])))
        self.attraction_points.append(Attractor(np.array([0.5, 0.5, -0.5])))
        self.attraction_points.append(Attractor(np.array([0.5, 0.5, -0.6])))
        self.attraction_points.append(Attractor(np.array([0.5, 0.5, -0.7])))
        self.attraction_points.append(Attractor(np.array([0.5, 0.5, -0.8])))

        # Create a set of nodes
        self.nodes = []
        self.root_node = Node(np.array([0.5, 0.5, -1]), parent=None)
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
        return self.kdtree.query(attractor.position,distance_upper_bound=self.attraction_dist)

    def update(self):
        # Clear all influence data
        for n in self.nodes:
            n.influenced_by = []
        for a in self.attraction_points:
            a.influencing_nodes = []

        # Calculate influence
        for a in self.attraction_points:
            d, ni = self.get_closest_node(a)
            if d != np.inf :
                n = self.nodes[ni]
                n.influenced_by.append(a)
                a.influencing_nodes.append(n)

        # Create new nodes
        fresh_nodes = []
        for n in self.nodes:
            if len(n.influenced_by) > 0:
                # Calculate direction of growth
                direction = np.array([.0,.0,.0])
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
        print(f"{len(fresh_nodes)} nodes added.")

        killables = []
        for a in self.attraction_points:
            if self.check_kill_distance(a):
                killables.append(a)
        for ka in killables:
            self.attraction_points.remove(ka)

        # Refresh KD-Tree
        self.build_spatial_index()


    def segment_traverse(self, node):
        #Leaf
        if len(node.children) == 0:
            return [node.position]
        #Segment
        if len(node.children) == 1:
            return [node.position] + self.segment_traverse(node.children[0])
        #Branch
        if len(node.children) >= 2:
            for cn in node.children:
                self.new_segment(cn)
            return [node.position]

    all_segments = []
    def new_segment(self,start_node):
        self.all_segments.append(self.segment_traverse(start_node))

    def visualize_gc(self):
        self.new_segment(self.root_node)
        print(f"no of segs: {len(self.all_segments)}")


    def visualize_with_spheres(self):
        # Define some colors
        color_dark = [0.1, 0.1, 0.1]
        color_red = [1, 0, 0]
        color_green = [0, 1, 0]
        color_blue = [0, 0, 1]

        # Drawing code
        all_drawables = []
        # all_drawables.append(draw_coordinate_box())

        # Draw attraction points
        for ap in self.attraction_points:
            if len(ap.influencing_nodes) > 0:
                all_drawables.append(draw_point(ap.position, color_blue))
            else:
                all_drawables.append(draw_point(ap.position, color_dark))

        # Draw nodes
        for n in self.nodes:
            if n.is_degen:
                all_drawables.append(draw_point(n.position, color_green))
            else:
                all_drawables.append(draw_point(n.position, color_red))

        return all_drawables


def draw_point(pt, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    s.translate(pt)
    return s


def draw_coordinate_box():
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def get_random_point_in_sphere() :
    u = random.random()
    x1 = random.random() - 0.5
    x2 = random.random() - 0.5
    x3 = random.random() - 0.5

    mag = math.sqrt(x1*x1 + x2*x2 + x3*x3)
    x1 /= mag
    x2 /= mag
    x3 /= mag

    c = u ** (1./3.)
    return np.array([x1*c, x2*c, x3*c]) + np.array([0.5,0.5,0.5])



if __name__ == "__main__":
    sca = SpaceColonization()

    def advance_key_callback(vis: o3d.visualization.Visualizer):
        vis.clear_geometries()
        sca.update()
        for geom in sca.visualize_with_spheres():
            vis.add_geometry(geom,reset_bounding_box=False)

    def advance2_key_callback(vis: o3d.visualization.Visualizer):
        vis.clear_geometries()
        for i in range(10):
            sca.update()
        for geom in sca.visualize_with_spheres():
            vis.add_geometry(geom,reset_bounding_box=False)

    def print_segs_callback(vis: o3d.visualization.Visualizer):
        sca.all_segments = []
        sca.visualize_gc()

    key_to_callback = {ord("A"): advance_key_callback, ord("S"): advance2_key_callback, ord("D"): print_segs_callback}
    o3d.visualization.draw_geometries_with_key_callbacks(sca.visualize_with_spheres(), key_to_callback)

import numpy as np
import open3d as o3d
import random
import math

from scipy.spatial import KDTree
from attractor import Attractor
from node import Node


class SpaceColonization:
    attraction_points: []
    nodes: []
    kdtree: KDTree

    # constant parameters
    attraction_dist: float = 0.6
    kill_distance: float = 0.05
    step_size: float = 0.02

    def __init__(self):
        # Seed Random
        random.seed(7)

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
        self.nodes.append(Node(np.array([0.5, 0.5, -1]), parent=None))

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
                direction = np.array([.0,.0,.0])
                atr: Attractor
                for atr in n.influenced_by:
                    direction += atr.position - n.position
                direction /= np.linalg.norm(direction)
                new_pos = n.position + direction * self.step_size
                fresh_nodes.append(Node(new_pos, parent=n))

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


    def get_drawables(self):
        # Define some colors
        color_dark = [0.1, 0.1, 0.1]
        color_red = [1, 0, 0]
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
        for geom in sca.get_drawables():
            vis.add_geometry(geom,reset_bounding_box=False)

    def advance2_key_callback(vis: o3d.visualization.Visualizer):
        vis.clear_geometries()
        for i in range(10):
            sca.update()
        for geom in sca.get_drawables():
            vis.add_geometry(geom,reset_bounding_box=False)

    key_to_callback = {ord("A"): advance_key_callback, ord("S"): advance2_key_callback}
    o3d.visualization.draw_geometries_with_key_callbacks(sca.get_drawables(), key_to_callback)

import numpy as np
import open3d as o3d
import random
import math

from scipy.spatial import KDTree
from sca_bud.attractor import Attractor
from sca_bud.node import Node
from graphics_helper import get_arrow, get_rotated_vector_by_angles, degree2rads, rotate_along_axis, \
    angle_between_two_vectors


class SpaceColonizationWithBuds:
    attraction_points: [Attractor]
    nodes: [Node]
    developable_nodes: [Node]
    root_node: Node
    # kdtree: KDTree
    developable_kdtree: KDTree

    # constant parameters
    attraction_dist: float = 0.3
    kill_distance: float = 0.05
    step_size: float = 0.01
    heading_w: float = 0.8
    angle_gamma: float = degree2rads(90)
    angle_beta: float = degree2rads(137.5)
    max_angle_deviation: float = degree2rads(20)
    max_branch_depth: int = 1

    gravity = np.array([0.0,0.0,-1.0])
    tropism_w: float = 0.0

    def __init__(self, attraction_points, root_pos, root_heading):
        # Create a set of attraction points
        self.attraction_points = attraction_points

        # Create a set of nodes
        self.nodes = []
        self.root_node = Node(root_pos, root_heading, parent=None)
        self.root_node.last_bud_heading = get_rotated_vector_by_angles(root_heading, self.angle_gamma, self.angle_beta)
        self.nodes.append(self.root_node)

        # Build spatial Index
        self.kdtree = None
        self.build_spatial_index()

    def build_spatial_index(self):
        # Create KD-Tree for nodes
        # nd = np.array([node.position for node in self.nodes])
        # self.kdtree = KDTree(nd)
        # Another KDtree for tip nodes and buds
        self.developable_nodes = []
        for node in self.nodes:
            if node.is_tip or node.is_bud:
                self.developable_nodes.append(node)
        dnd = np.array([dnode.position for dnode in self.developable_nodes])
        self.developable_kdtree = KDTree(dnd)

    def check_kill_distance(self, attractor):
        return len(self.developable_kdtree.query_ball_point(attractor.position, r=self.kill_distance)) > 0

    def get_closest_node(self, attractor):
        return self.developable_kdtree.query(attractor.position, distance_upper_bound=self.attraction_dist)

    def get_bud_freq(self, branch_depth):
        if branch_depth == 0:
            return 5
        elif branch_depth == 1:
            return 3
        else:
            return 2


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
                n = self.developable_nodes[ni]
                n.influenced_by.append(a)
                a.influencing_nodes.append(n)

        # Create new nodes
        fresh_nodes = []
        for n in self.nodes:
            if n.is_bud and not n.is_tip:
                if len(n.influenced_by) > 0:
                    direction = np.array([.0, .0, .0])
                    no_of_valid_attr = 0
                    atr: Attractor
                    for atr in n.influenced_by:
                        vec_to_attr = atr.position - n.position
                        # vec_to_attr /= np.linalg.norm(vec_to_attr)
                        d_angle = angle_between_two_vectors(n.bud_heading, vec_to_attr)
                        if d_angle > self.max_angle_deviation:
                            continue
                        direction += vec_to_attr
                        no_of_valid_attr += 1

                    if no_of_valid_attr == 0:
                        continue

                    if self.max_branch_depth != -1 and n.branch_depth == self.max_branch_depth:
                        continue

                    # if n.dist_to_bud / n.dist_to_root > (1 / (n.branch_depth + 1)):
                    #     continue

                    direction /= np.linalg.norm(direction)
                    dir2 = self.heading_w * n.bud_heading + (1 - self.heading_w) * direction
                    dir3 = self.tropism_w * self.gravity + (1 - self.tropism_w) * dir2
                    dir3 /= np.linalg.norm(dir3)

                    new_pos = n.position + dir3 * self.step_size
                    new_node = Node(new_pos, heading=dir2, parent=n)

                    new_node.last_bud_heading = get_rotated_vector_by_angles(dir2, self.angle_gamma, self.angle_beta)
                    new_node.branch_depth += 1
                    new_node.dist_to_bud = 1

                    # Set as child in n
                    n.children.append(new_node)
                    fresh_nodes.append(new_node)

                    # Node is no longer bud
                    n.is_bud = False

            if n.is_tip:
                if len(n.influenced_by) > 0:
                    direction = np.array([.0, .0, .0])
                    no_of_valid_attr = 0
                    atr: Attractor
                    for atr in n.influenced_by:
                        vec_to_attr = atr.position - n.position
                        # vec_to_attr /= np.linalg.norm(vec_to_attr)
                        d_angle = angle_between_two_vectors(n.heading, vec_to_attr)
                        if d_angle > self.max_angle_deviation:
                            continue
                        direction += vec_to_attr
                        no_of_valid_attr += 1

                    if no_of_valid_attr == 0:
                        continue

                    # Special degen case
                    # if len(n.influenced_by) == 2:
                    #     center = (n.influenced_by[0].position + n.influenced_by[1].position) / 2
                    #     if np.linalg.norm(n.position - center) < self.step_size:
                    #         direction = n.influenced_by[0].position - n.position

                    direction /= np.linalg.norm(direction)
                    dir2 = self.heading_w * n.heading + (1 - self.heading_w) * direction

                    new_pos = n.position + dir2 * self.step_size
                    new_node = Node(new_pos, heading=dir2, parent=n)
                    if new_node.dist_to_bud % self.get_bud_freq(n.branch_depth) == 0:
                        new_node.is_bud = True
                        rm = rotate_along_axis(n.last_bud_heading, n.heading, self.angle_beta)
                        new_node.bud_heading = rm
                        new_node.last_bud_heading = rm
                    # Set as child in n
                    n.children.append(new_node)
                    fresh_nodes.append(new_node)

                    # Node is no longer tip
                    n.is_tip = False

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
                all_segments.append(parent_element + segment_traverse(start_node))

        new_segment(self.root_node, parent_element=None)

        return all_segments

    def draw_point(self, pt, color):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        s.paint_uniform_color(color)
        s.compute_vertex_normals()
        s.translate(pt)
        return s

    def visualize_with_spheres(self, draw_buds=True, draw_attractors=True):
        # Define some colors
        color_dark = [0.1, 0.1, 0.1]
        color_red = [1, 0, 0]
        color_green = [0, 1, 0]
        color_yellow = [1, 1, 0]
        color_blue = [0, 0, 1]

        # Drawing code
        all_drawables = []
        # all_drawables.append(draw_coordinate_box())

        # Draw attraction points
        if draw_attractors:
            for ap in self.attraction_points:
                if len(ap.influencing_nodes) > 0:
                    all_drawables.append(self.draw_point(ap.position, color_blue))
                else:
                    all_drawables.append(self.draw_point(ap.position, color_dark))

        # Draw nodes
        for n in self.nodes:
            if n.is_tip:
                # all_drawables.append(get_arrow(n.position,vec=n.heading, color=color_green))
                all_drawables.append(self.draw_point(n.position, color_green))
            elif n.is_bud:
                if draw_buds:
                    all_drawables.append(get_arrow(n.position, vec=n.bud_heading, color=color_yellow))
                    all_drawables.append(self.draw_point(n.position, color_yellow))
                else:
                    all_drawables.append(self.draw_point(n.position, color_red))
            else:
                all_drawables.append(self.draw_point(n.position, color_red))

        return all_drawables


if __name__ == "__main__":

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


    print("Running SCA with Buds variation ...")

    attraction_pts = []
    n_attraction_pts = 600
    for i in range(n_attraction_pts):
        rpos = get_random_point_in_sphere()
        attraction_pts.append(Attractor(rpos))
    attraction_pts.append(Attractor(np.array([0.5, 0.5, -0.4])))
    attraction_pts.append(Attractor(np.array([0.5, 0.5, -0.5])))
    attraction_pts.append(Attractor(np.array([0.5, 0.5, -0.6])))
    attraction_pts.append(Attractor(np.array([0.5, 0.5, -0.7])))
    attraction_pts.append(Attractor(np.array([0.5, 0.5, -0.8])))

    root_pos = np.array([0.5, 0.5, -1])
    root_heading = np.array([0, 0, 1])

    scab = SpaceColonizationWithBuds(attraction_pts, root_pos, root_heading)


    def advance_key_callback(vis: o3d.visualization.Visualizer):
        vis.clear_geometries()
        scab.update()
        for geom in scab.visualize_with_spheres():
            vis.add_geometry(geom, reset_bounding_box=False)


    def run_key_callback(vis: o3d.visualization.Visualizer):
        vis.clear_geometries()
        scab.run()
        for geom in scab.visualize_with_spheres():
            vis.add_geometry(geom, reset_bounding_box=False)


    key_to_callback = {ord("A"): advance_key_callback, ord("D"): run_key_callback}
    o3d.visualization.draw_geometries_with_key_callbacks(scab.visualize_with_spheres(), key_to_callback)

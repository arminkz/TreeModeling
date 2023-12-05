import numpy as np
import open3d as o3d
import laspy
import random
import math


from sca.attractor import Attractor
from sca.space_colonization import SpaceColonization
from pf.particle_flow import ParticleFlow
from sca_bud.space_colonization_bud import SpaceColonizationWithBuds

# Seed Random
random.seed(7)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def get_random_point_in_sphere(radius):
    u = random.random()
    x1 = random.random() - 0.5
    x2 = random.random() - 0.5
    x3 = random.random() - 0.5

    mag = math.sqrt(x1 * x1 + x2 * x2 + x3 * x3)
    x1 /= mag
    x2 /= mag
    x3 /= mag

    c = u ** (1. / 3.)
    return np.array([x1 * c, x2 * c, x3 * c]) + np.array([0, 0, 0.5])


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def generate_sweep_surface(segment):
    """
    Given a segment of the skeleton partial mesh for that part is generated
    :param segment:
    :return:
    """
    vertices = []
    faces = []
    print(f"Segment has {len(segment)} points rad: {segment[0][0]}")
    for cpi in range(len(segment) - 1):
        sweep_radius = segment[cpi][0] * 0.01
        p1 = np.array(segment[cpi][1])
        p2 = np.array(segment[cpi + 1][1])
        p3 = np.array([0, 0, 0])
        if cpi < len(segment) - 2:
            p3 = np.array(segment[cpi + 2][1])

        # Trajectory direction
        d = p2 - p1
        d2 = None
        if (p3 == np.array([0, 0, 0])).all():
            d2 = d
        else:
            d2 = p3 - p2

        # Prependicular 1
        b = np.cross(d, np.array([2, 3, 4]))
        b = (b / np.linalg.norm(b)) * sweep_radius

        b2 = np.cross(d2, np.array([2, 3, 4]))
        b2 = (b2 / np.linalg.norm(b2)) * sweep_radius

        # Points round the circle
        angles = np.arange(0, 2 * math.pi, 0.2)
        angles = np.append(angles, angles[0]) #make it circular
        for angle_i in range(len(angles) - 1):
            angle = angles[angle_i]
            angle2 = angles[angle_i + 1]
            rot_s = rotation_matrix(d, angle)
            rot_s2 = rotation_matrix(d, angle2)
            rot_q = rotation_matrix(d2, angle)
            rot_q2 = rotation_matrix(d2, angle2)
            s1 = rot_s @ b
            s2 = rot_s2 @ b
            q1 = rot_q @ b2
            q2 = rot_q2 @ b2
            c1p1 = p1 + s1
            c2p1 = p1 + s2
            c1p2 = p2 + q1
            c2p2 = p2 + q2

            vcount = len(vertices)

            vertices.append(c1p1)
            vertices.append(c2p1)
            vertices.append(c1p2)
            vertices.append(c2p2)

            face1 = [0 + vcount, 1 + vcount, 2 + vcount]
            face2 = [1 + vcount, 3 + vcount, 2 + vcount]
            face3 = [2 + vcount, 1 + vcount, 0 + vcount]
            face4 = [2 + vcount, 3 + vcount, 1 + vcount]

            faces.append(face1)
            faces.append(face2)
            faces.append(face3)
            faces.append(face4)

    sweep_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),
                                           triangles=o3d.utility.Vector3iVector(faces))
    sweep_mesh.paint_uniform_color([135 / 256, 62 / 256, 35 / 256])
    return sweep_mesh

# Generate Random Points
# attraction_pts = []
# n_attraction_pts = 600
# for i in range(n_attraction_pts):
#     rpos = get_random_point_in_sphere()
#     attraction_pts.append(Attractor(rpos))
# attraction_pts.append(Attractor(np.array([0.5, 0.5, -0.4])))
# attraction_pts.append(Attractor(np.array([0.5, 0.5, -0.5])))
# attraction_pts.append(Attractor(np.array([0.5, 0.5, -0.6])))
# attraction_pts.append(Attractor(np.array([0.5, 0.5, -0.7])))
# attraction_pts.append(Attractor(np.array([0.5, 0.5, -0.8])))

# Root
# root_pos = np.array([0.5, 0.5, -1])

# print("running SCA...")
# sca = SpaceColonization(attraction_pts, root_pos)
# sca.run()
# skeleton = sca.get_skeleton()
#
# all_sweep_meshes = []
# for seg in skeleton:
#     print(seg)
#     m = generate_sweep_surface(seg)
#     all_sweep_meshes.append(m)
# all_sweep_meshes.append(generate_sweep_surface(skeleton[0]))
# o3d.visualization.draw_geometries(all_sweep_meshes)

# initial_positions = []
# n_inner_pts = 100
# for i in range(n_inner_pts):
#     rpos = get_random_point_in_sphere()
#     initial_positions.append(rpos)
#
# pf = ParticleFlow(initial_positions,root_pos)
# pf.run()
# skeleton = pf.get_skeleton()
#
# all_sweep_meshes = []
# for seg in skeleton:
#     print(seg)
#     m = generate_sweep_surface(seg)
#     all_sweep_meshes.append(m)
# # all_sweep_meshes.append(generate_sweep_surface(skeleton[0]))
# o3d.visualization.draw_geometries(all_sweep_meshes)


# Load LAS file
las = laspy.read("isolated_tree_las/bc_sample_tree.las")

print(las.header)
print(las.header.point_format)
print(f"Point count : {las.header.point_count} ")
print(las.vlrs)
print(list(las.point_format.dimension_names))

# Normalize Z
z_r = (np.max(las.Z) - np.min(las.Z))
min_z = np.min(las.Z)
las.Z -= min_z
 # / z_r

# Normalize X,Y
x_r = (np.max(las.X) - np.min(las.X))
y_r = np.max(las.Y) - np.min(las.Y)
las.X -= np.mean(las.X, dtype="int32")
las.Y -= np.mean(las.Y, dtype="int32")


normalizedX = las.X.astype(dtype="float64") / x_r
normalizedY = las.Y.astype(dtype="float64") / x_r
normalizedZ = las.Z.astype(dtype="float64") / x_r

print(f"X {np.min(normalizedX)} {np.max(normalizedX)}")
print(f"Y {np.min(normalizedY)} {np.max(normalizedY)}")
print(f"Z {np.min(normalizedZ)} {np.max(normalizedZ)}")

point_data = np.stack([normalizedX, normalizedY, normalizedZ], axis=0).transpose((1, 0))
color_data = np.stack([las.red, las.green, las.blue], axis=0).transpose((1, 0)) / 65024

geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(point_data)
geom.colors = o3d.utility.Vector3dVector(color_data)

# o3d.visualization.draw_geometries([geom])
hull, _ = geom.compute_convex_hull()
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hull))

print(point_data.shape)

attraction_pts = []
for i in range(0,point_data.shape[0],2):
    attraction_pts.append(Attractor(point_data[i]))

# Add points inside
points_inside_sphere = []
for i in range(2000):
    points_inside_sphere.append(get_random_point_in_sphere(0.5))

occ_test = scene.compute_occupancy(np.array(points_inside_sphere, dtype='float32'))
for i in range(2000):
    if occ_test[i] == 1.0:
        attraction_pts.append(Attractor(points_inside_sphere[i]))

# Add mid axis
for i in np.linspace(0,1,30):
    attraction_pts.append(Attractor(np.array([0, 0, i])))
#
# attraction_pts.append(Attractor(np.array([0, 0, -0.5])))
# attraction_pts.append(Attractor(np.array([0, 0, -0.6])))
# attraction_pts.append(Attractor(np.array([0, 0, -0.7])))
# attraction_pts.append(Attractor(np.array([0, 0, -0.8])))

root_pos = np.array([0,0,-0.1])
root_heading = np.array([0.0,0.0,1.0])
print("Running SCA...")
scab = SpaceColonizationWithBuds(attraction_pts, root_pos, root_heading)
# sca.run()
# skeleton = sca.get_skeleton()
# #
# all_sweep_meshes = []
# for seg in skeleton:
#     print(seg)
#     m = generate_sweep_surface(seg)
#     all_sweep_meshes.append(m)
# # all_sweep_meshes.append(generate_sweep_surface(skeleton[0]))
# o3d.visualization.draw_geometries(all_sweep_meshes)

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
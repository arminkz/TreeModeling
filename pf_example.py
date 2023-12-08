import numpy as np
import open3d as o3d
import laspy
import random
import math
from pf.particle_flow import ParticleFlow

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

root_pos = np.array([0.5, 0.5, -1])
pf = ParticleFlow(point_data, root_pos)

step = 1

def advance_key_callback(vis: o3d.visualization.Visualizer):
    global step
    vis.clear_geometries()
    pf.update()
    for geom in pf.visualize_with_spheres():
        vis.add_geometry(geom, reset_bounding_box=False)
    vis.capture_screen_image(f"captures/pf/{step}.png", do_render=True)
    step += 1


def run_key_callback(vis: o3d.visualization.Visualizer):
    vis.clear_geometries()
    pf.run()
    for geom in pf.visualize_with_spheres():
        vis.add_geometry(geom, reset_bounding_box=False)


key_to_callback = {ord("A"): advance_key_callback, ord("D"): run_key_callback}
o3d.visualization.draw_geometries_with_key_callbacks(pf.visualize_with_spheres(), key_to_callback)
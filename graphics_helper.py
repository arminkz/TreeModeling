import open3d as o3d
import numpy as np
import math
from scipy.spatial.transform.rotation import Rotation as R


def create_arrow(scale=0.5):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=0.01,
        cone_height=0.02,
        cylinder_radius=0.005,
        cylinder_height=0.02)
    return(mesh_frame)


def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec ():
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def get_arrow(origin=[0, 0, 0], end=None, vec=None, color=[1, 0, 0]):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow()
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)

    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    return(mesh)


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the
    z axis vector of the original FOR. The first rotation that is
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis.

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec ():
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan2(vec[1],vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1);
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0]/vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)


def get_rotated_vector_by_angles(vec, gamma, beta):

    r = np.array([2,3,4], dtype='float64')

    c1 = np.cross(vec,r)
    c1 /= np.linalg.norm(c1)

    c2 = np.cross(vec, c1)
    c2 /= np.linalg.norm(c2)

    # Rotate by gamma
    Rz = rotation_matrix(c1, gamma)
    Ry = rotation_matrix(c2, beta)

    return Rz @ Ry @ vec


def degree2rads(degree):
    return (degree / 180) * math.pi


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


def rotate_along_axis(vector, axis, theta):
    axis /= np.linalg.norm(axis)
    axis *= theta
    rot = R.from_rotvec(axis)
    return rot.apply(vector)


def angle_between_two_vectors(a, b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return np.arccos(np.dot(a, b))

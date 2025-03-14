import numpy as np
from scipy.spatial.transform import Rotation as R

def translate(x, y, z):
    '''
    return Homogeneous coordinate.
    :param x:
    :param y:
    :param z:
    :return:
    '''
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

def rotate_x(theta):
    return np.array([[1,             0,              0,             0],
                     [0,  np.cos(theta),  -np.sin(theta),           0],
                     [0,  np.sin(theta),   np.cos(theta),           0],
                     [0,             0,              0,             1]])

def rotate_y(theta):
    return np.array([[ np.cos(theta),  0,  np.sin(theta),  0],
                     [             0,  1,              0,  0],
                     [-np.sin(theta),  0,  np.cos(theta),  0],
                     [             0,  0,              0,  1]])

def rotate_z(theta):
    return np.array([[ np.cos(theta), -np.sin(theta),  0,  0],
                     [ np.sin(theta),  np.cos(theta),  0,  0],
                     [             0,              0,  1,  0],
                     [             0,              0,  0,  1]])

def compute_global_pose(node):
    """Recursively compute the global transform of 'node' by following its parents up to the root."""
    if node['parent'] is None:
        # This is the root node; global transform is just its local transform
        return node['transform']
    else:
        parent_tf = compute_global_pose(node['parent'])
        return parent_tf @ node['transform']

def tranform_vector(T, p):
    '''
    Transform 3d vector with homogeneous transformation.
    :param T: Homogeneous transform.
    :param p: 3d vector to be transformed.
    :return: Transformed 3d vector
    '''
    p_aug = np.concatenate((p, np.array([0])))
    p_aug = T@p
    return np.array([p_aug[0]/p_aug[3], p_aug[1]/p_aug[3], p_aug[2]/p_aug[3]])

def rotation_to_zyx_euler(Rmat):
    '''
    Given rotation matrix, return yaw, pitch, roll angles.
    :param R:
    :return:
    '''
    r = R.from_matrix(Rmat)
    return r.as_euler('ZYX')
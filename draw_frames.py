import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting in older matplotlib versions
import numpy as np
from quadruped import Quadruped
from utility import compute_global_pose

def draw_kinematic_tree(root_node, ax=None):
    """
    Draws each node's position as a point in 3D, plus a small set of axes for that frame.
    Also draws a line from each child back to its parent.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # We can do BFS or DFS; here is BFS for clarity
    queue = [root_node]

    while queue:
        node = queue.pop(0)

        # Compute this node's global transform
        node_tf = compute_global_pose(node)
        # Extract the origin of this node
        px, py, pz = node_tf[0,3], node_tf[1,3], node_tf[2,3]

        # Plot the node's origin as a scatter point
        ax.scatter(px, py, pz)

        # Draw the 3 coordinate axes of this node (x, y, z)
        # Let's define a small axis length
        axis_len = 0.05
        # Each axis direction is in the columns [0:3] of the rotation part
        x_axis = node_tf[0:3, 0]
        y_axis = node_tf[0:3, 1]
        z_axis = node_tf[0:3, 2]

        # We'll draw a line for each axis (X in red-ish, Y in green-ish, Z in blue-ish, etc.)
        # (Though the user didn't explicitly request color, it's typical to differentiate axes.)
        # start = (px, py, pz)
        # endX = start + axis_len * x_axis, etc.

        ax.plot(
            [px, px + axis_len * x_axis[0]],
            [py, py + axis_len * x_axis[1]],
            [pz, pz + axis_len * x_axis[2]],
            color='red',
            linewidth=3
        )

        ax.plot(
            [px, px + axis_len * y_axis[0]],
            [py, py + axis_len * y_axis[1]],
            [pz, pz + axis_len * y_axis[2]],
            color='yellow',
            linewidth=3
        )

        ax.plot(
            [px, px + axis_len * z_axis[0]],
            [py, py + axis_len * z_axis[1]],
            [pz, pz + axis_len * z_axis[2]],
            color='blue',
            linewidth=3
        )

        # If not the root, draw line from this node to its parent's origin
        parent = node['parent']
        if parent is not None:
            parent_tf = compute_global_pose(parent)
            ppx, ppy, ppz = parent_tf[0,3], parent_tf[1,3], parent_tf[2,3]
            # Draw a line from parent -> this node
            ax.plot([px, ppx], [py, ppy], [pz, ppz], linestyle='-', color='black')

        # Enqueue children
        for child_name, child_node in node['children'].items():
            queue.append(child_node)

    # Optionally set some axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot (you can remove this or control it outside the function if desired)
    plt.show()

if __name__=="__main__":
    quad = Quadruped()
    joints = quad.get_joint_angles()
    joints['FLS1'] = -0.1
    joints['FLS2'] = -0.3
    joints['FLE'] = 0.6
    joints['FRS1'] = 0.1
    joints['FRS2'] = -0.3
    joints['FRE'] = 0.6
    joints['RLS1'] = -0.1
    joints['RLS2'] = -0.3
    joints['RLE'] = 0.6
    joints['RRS1'] = 0.1
    joints['RRS2'] = -0.3
    joints['RRE'] = 0.6
    quad.set_joint_angles(joints)
    root = quad.get_KTtree()
    draw_kinematic_tree(root)
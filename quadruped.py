import numpy as np
from utility import *
from functools import reduce

class Quadruped:
    def __init__(self, Lf=0.3, Lr=0.3, Ls=0.2, Lss=0.1, Lleg=0.3, Lfoot=0.3):
        self.Lf = Lf
        self.Lr = Lr
        self.Ls = Ls
        self.Lss = Lss
        self.Lleg = Lleg
        self.Lfoot = Lfoot
        # Joints are list of angles.
        # [front left joints, front right joints, rear left joints, rear right joints]
        # each group, i.e front left joints, contains 3 joints.
        self.joints = {
            'FLS1': 0,
            'FLS2': 0,
            'FLE': 0,
            'FRS1': 0,
            'FRS2': 0,
            'FRE': 0,
            'RLS1': 0,
            'RLS2': 0,
            'RLE': 0,
            'RRS1': 0,
            'RRS2': 0,
            'RRE': 0
        }
        self.rotation_axes = [
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([1, 0, 0])
        ]

    def set_joint_angles(self, joints):
        if not len(joints) == 12:
            raise("Joints number should be 12")
        self.joints = joints

    def get_joint_angles(self):
        return self.joints

    def get_foot_tip_jacobian(self):
        '''
        Get jacobian matrices from joints of each branch to top of each foot.
        It is expressed in body frame.
        :return: List of 4 jacobian matrices.
        '''

        def get_transforms_list(node):
            transforms = []
            it = node
            while True:
                transforms.append(it['transform'])
                # If there are no children, we've reached the foot node
                if not it['children']:
                    break
                # Otherwise, move to the (only) child
                it = list(it['children'].values())[0]
            return transforms

        body_frame = self.get_KTtree()
        FL_transforms = get_transforms_list(body_frame['children']['FL_shoulder_1'])
        FR_transforms = get_transforms_list(body_frame['children']['FR_shoulder_1'])
        RL_transforms = get_transforms_list(body_frame['children']['RL_shoulder_1'])
        RR_transforms = get_transforms_list(body_frame['children']['RR_shoulder_1'])

        J_FL = self.get_jacobian(FL_transforms, self.rotation_axes[0:3])
        J_FR = self.get_jacobian(FR_transforms, self.rotation_axes[3:6])
        J_RL = self.get_jacobian(RL_transforms, self.rotation_axes[6:9])
        J_RR = self.get_jacobian(RR_transforms, self.rotation_axes[9:12])

        return [J_FL, J_FR, J_RL, J_RR]

    def get_jacobian(self, transformations, rotation_axes):
        T_bi = np.identity(4)
        J_list = []

        # This multiplies all transforms in 'transformations' to get the base->EE transform.
        # So T_ee is the transform that takes a point in end-effector coords to base coords.
        T_ee = reduce(lambda A, B: A @ B, transformations, np.eye(4))

        # The end-effector's position in base frame:
        P_ee = T_ee[0:3, 3]

        for i in range(len(rotation_axes)):
            # Multiply up to the i-th joint to get base->joint_i transform
            T_bi = T_bi @ transformations[i]

            # Joint i's origin in the base frame:
            P_bi = T_bi[0:3, 3]

            # Vector from joint i's origin to end-effector
            P_ie = P_ee - P_bi

            # 'rotation_axes[i]' is presumably the joint axis in joint i's local frame.
            # Multiply it by T_bi's rotation part to express it in the base frame:
            w = T_bi[0:3, 0:3] @ rotation_axes[i]

            # For a revolute joint:
            # linear velocity contribution =  w x (P_ee - P_bi)
            # angular velocity contribution = w
            # (But see note below about ordering.)
            J = np.concatenate((w, np.cross(w, P_ie)))

            J_list.append(J)

    def get_KTtree(self):
        """
        Build a recursive Kinematic Tree (KT) where each node has:
        {
            'name': <string>,
            'parent': <reference to parent node or None if root>,
            'transform': <4x4 local transform>,
            'children': {
                <child_name>: <child_node>,
                ...
            }
        }
        """
        # Body node (root of the tree)
        body = {
            'name': 'body',
            'transform': np.eye(4),  # Root node: identity
            'parent': None,          # No parent
            'children': {}
        }

        # -- FRONT LEFT LEG -- #
        # Shoulder 1 node
        FL_s1 = {
            'name': 'FL_shoulder_1',
            'transform': (
                translate(-self.Ls, self.Lf, 0) @
                rotate_y(self.joints['FLS1'])
            ),
            'parent': body,  # reference the parent node
            'children': {}
        }
        body['children']['FL_shoulder_1'] = FL_s1

        # Shoulder 2 node
        FL_s2 = {
            'name': 'FL_shoulder_2',
            'transform': (
                translate(-self.Lss, 0, 0) @
                rotate_x(self.joints['FLS2'] - np.pi/2)
            ),
            'parent': FL_s1,
            'children': {}
        }
        FL_s1['children']['FL_shoulder_2'] = FL_s2

        # Elbow node
        FL_elbow = {
            'name': 'FL_elbow',
            'transform': (
                translate(0, self.Lleg, 0) @
                rotate_x(self.joints['FLE'])
            ),
            'parent': FL_s2,
            'children': {}
        }
        FL_s2['children']['FL_elbow'] = FL_elbow

        # Foot node (child of elbow)
        FL_foot = {
            'name': 'FL_foot',
            'transform': translate(0, self.Lfoot, 0),
            'parent': FL_elbow,
            'children': {}
        }
        FL_elbow['children']['FL_foot'] = FL_foot

        # -- FRONT RIGHT LEG -- #
        FR_s1 = {
            'name': 'FR_shoulder_1',
            'transform': (
                translate(self.Ls, self.Lf, 0) @
                rotate_y(self.joints['FRS1'])
            ),
            'parent': body,
            'children': {}
        }
        body['children']['FR_shoulder_1'] = FR_s1

        FR_s2 = {
            'name': 'FR_shoulder_2',
            'transform': (
                translate(self.Lss, 0, 0) @
                rotate_x(self.joints['FRS2'] - np.pi/2)
            ),
            'parent': FR_s1,
            'children': {}
        }
        FR_s1['children']['FR_shoulder_2'] = FR_s2

        FR_elbow = {
            'name': 'FR_elbow',
            'transform': (
                translate(0, self.Lleg, 0) @
                rotate_x(self.joints['FRE'])
            ),
            'parent': FR_s2,
            'children': {}
        }
        FR_s2['children']['FR_elbow'] = FR_elbow

        FR_foot = {
            'name': 'FR_foot',
            'transform': translate(0, self.Lfoot, 0),
            'parent': FR_elbow,
            'children': {}
        }
        FR_elbow['children']['FR_foot'] = FR_foot

        # -- REAR LEFT LEG -- #
        RL_s1 = {
            'name': 'RL_shoulder_1',
            'transform': (
                translate(-self.Ls, -self.Lr, 0) @
                rotate_y(self.joints['RLS1'])
            ),
            'parent': body,
            'children': {}
        }
        body['children']['RL_shoulder_1'] = RL_s1

        RL_s2 = {
            'name': 'RL_shoulder_2',
            'transform': (
                translate(-self.Lss, 0, 0) @
                rotate_x(self.joints['RLS2'] - np.pi/2)
            ),
            'parent': RL_s1,
            'children': {}
        }
        RL_s1['children']['RL_shoulder_2'] = RL_s2

        RL_elbow = {
            'name': 'RL_elbow',
            'transform': (
                translate(0, self.Lleg, 0) @
                rotate_x(self.joints['RLE'])
            ),
            'parent': RL_s2,
            'children': {}
        }
        RL_s2['children']['RL_elbow'] = RL_elbow

        RL_foot = {
            'name': 'RL_foot',
            'transform': translate(0, self.Lfoot, 0),
            'parent': RL_elbow,
            'children': {}
        }
        RL_elbow['children']['RL_foot'] = RL_foot

        # -- REAR RIGHT LEG -- #
        RR_s1 = {
            'name': 'RR_shoulder_1',
            'transform': (
                translate(self.Ls, -self.Lr, 0) @
                rotate_y(self.joints['RRS1'])
            ),
            'parent': body,
            'children': {}
        }
        body['children']['RR_shoulder_1'] = RR_s1

        RR_s2 = {
            'name': 'RR_shoulder_2',
            'transform': (
                translate(self.Lss, 0, 0) @
                rotate_x(self.joints['RRS2'] - np.pi/2)
            ),
            'parent': RR_s1,
            'children': {}
        }
        RR_s1['children']['RR_shoulder_2'] = RR_s2

        RR_elbow = {
            'name': 'RR_elbow',
            'transform': (
                translate(0, self.Lleg, 0) @
                rotate_x(self.joints['RRE'])
            ),
            'parent': RR_s2,
            'children': {}
        }
        RR_s2['children']['RR_elbow'] = RR_elbow

        RR_foot = {
            'name': 'RR_foot',
            'transform': translate(0, self.Lfoot, 0),
            'parent': RR_elbow,
            'children': {}
        }
        RR_elbow['children']['RR_foot'] = RR_foot

        # Return the root of the entire kinematic tree
        return body
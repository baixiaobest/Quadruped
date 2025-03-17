import numpy as np
from utility import *
from functools import reduce
from optimize import foot_tip_IK_LM

class Quadruped:
    def __init__(self, Lf=0.3, Lr=0.3, Ls=0.2, Lss=0.1, Lleg=0.3, Lfoot=0.3, joint_angles_min=[-np.pi/2, -np.pi/2, -np.pi/2], joint_angles_max=[np.pi/2, np.pi/2, np.pi/2]):
        self.Lf = Lf
        self.Lr = Lr
        self.Ls = Ls
        self.Lss = Lss
        self.Lleg = Lleg
        self.Lfoot = Lfoot
        self.joint_angles_min = joint_angles_min
        self.joint_angles_max = joint_angles_max
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
            'FL1': 0,
            'FL2': 0,
            'FL3': 0,
            'FR1': 0,
            'FR2': 0,
            'FR3': 0,
            'RL1': 0,
            'RL2': 0,
            'RL3': 0,
            'RR1': 0,
            'RR2': 0,
            'RR3': 0
        }
        self.rotation_axes = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
        ]
        self.joint_angles_min = joint_angles_min
        self.joint_angles_max = joint_angles_max

        self.body_transform = np.eye(4)

    def set_joint_angles(self, joints):
        for joint in joints:
            if joint not in self.joints:
                raise(Exception(f"Joint {joint} is not valid."))
            if joints[joint] < self.joint_angles_min[0] or joints[joint] > self.joint_angles_max[0]:
                raise(Exception(f"Joint {joint} is out of bounds."))
            self.joints[joint] = joints[joint]

    def get_joint_angles(self):
        return self.joints

    def set_body_tranform(self, T):
        if not (T.shape[0] == 4 and T.shape[1] == 4):
            raise(Exception("Transform should be 4x4"))
        self.body_transform = T

    def get_transforms_list(self, node):
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
    
    def get_foot_tip_transforms(self):
        '''
        Get the transformation matrices from body to foot tip.
        :return: List of 4 transformation matrices.
        '''
        world = self.get_KTtree()
        body_frame = world['children']['body']
        FL_transforms = self.get_transforms_list(body_frame['children']['FL1'])
        FR_transforms = self.get_transforms_list(body_frame['children']['FR1'])
        RL_transforms = self.get_transforms_list(body_frame['children']['RL1'])
        RR_transforms = self.get_transforms_list(body_frame['children']['RR1'])

        FL_foot_tip = self.body_transform @ reduce(lambda A, B: A @ B, FL_transforms, np.eye(4))
        FR_foot_tip = self.body_transform @ reduce(lambda A, B: A @ B, FR_transforms, np.eye(4))
        RL_foot_tip = self.body_transform @ reduce(lambda A, B: A @ B, RL_transforms, np.eye(4))
        RR_foot_tip = self.body_transform @ reduce(lambda A, B: A @ B, RR_transforms, np.eye(4))

        return FL_foot_tip, FR_foot_tip, RL_foot_tip, RR_foot_tip

    def get_foot_tip_jacobians(self):
        '''
        Get jacobian matrices from joints of each branch to top of each foot.
        It is expressed in body frame.
        :return: List of 4 jacobian matrices.
        '''

        world = self.get_KTtree()
        body_frame = world['children']['body']
        FL_transforms = self.get_transforms_list(body_frame['children']['FL1'])
        FR_transforms = self.get_transforms_list(body_frame['children']['FR1'])
        RL_transforms = self.get_transforms_list(body_frame['children']['RL1'])
        RR_transforms = self.get_transforms_list(body_frame['children']['RR1'])

        J_FL = self.get_foot_tip_jacobian(body_frame['transform'], FL_transforms, self.rotation_axes[0:3])
        J_FR = self.get_foot_tip_jacobian(body_frame['transform'], FR_transforms, self.rotation_axes[3:6])
        J_RL = self.get_foot_tip_jacobian(body_frame['transform'], RL_transforms, self.rotation_axes[6:9])
        J_RR = self.get_foot_tip_jacobian(body_frame['transform'], RR_transforms, self.rotation_axes[9:12])

        return J_FL, J_FR, J_RL, J_RR

    def get_foot_tip_jacobian(self, body_transform, transformations, rotation_axes):
        '''
        The foot tip Jacobian of one of the foot.
        :param body_transform:
        :param transformations:
        :param rotation_axes:
        :return:
        '''
        T_i = body_transform
        J_list = []

        # So T_ee is the transform that takes a point in end-effector frame to world frame.
        T_ee = body_transform @ reduce(lambda A, B: A @ B, transformations, np.eye(4))

        # The end-effector's position in base frame:
        P_ee = T_ee[0:3, 3]

        for i in range(len(rotation_axes)):
            # Multiply up to the i-th joint to get base->joint_i transform
            T_i = T_i @ transformations[i]

            # Joint i's origin in the world frame:
            P_i = T_i[0:3, 3]

            # Vector from joint i's origin to end-effector
            P_ie = P_ee - P_i

            # 'rotation_axes[i]' is the joint axis in joint i's joint frame.
            # Multiply it by T_wi's rotation part to express it in the world frame:
            w = T_i[0:3, 0:3] @ rotation_axes[i]

            J = np.concatenate((w, np.cross(w, P_ie)))

            J_list.append(J)

        return np.column_stack(J_list)

    def get_body_jacobians(self):
        '''
        Get body Jacobains (actuated joints + unactuated joints) from all 4 legs.
        :return:
        '''
        world = self.get_KTtree()
        body_frame = world['children']['body']
        T_b = body_frame['transform']
        FL_transforms = self.get_transforms_list(body_frame['children']['FL1'])
        FR_transforms = self.get_transforms_list(body_frame['children']['FR1'])
        RL_transforms = self.get_transforms_list(body_frame['children']['RL1'])
        RR_transforms = self.get_transforms_list(body_frame['children']['RR1'])

        FL_foot_tip = T_b @ reduce(lambda A, B: A @ B, FL_transforms, np.eye(4))
        FR_foot_tip = T_b @ reduce(lambda A, B: A @ B, FR_transforms, np.eye(4))
        RL_foot_tip = T_b @ reduce(lambda A, B: A @ B, RL_transforms, np.eye(4))
        RR_foot_tip = T_b @ reduce(lambda A, B: A @ B, RR_transforms, np.eye(4))

        FL_foot_yaw, FL_foot_pitch, FL_foot_roll = rotation_to_zyx_euler(FL_foot_tip[0:3, 0:3])
        FR_foot_yaw, FR_foot_pitch, FR_foot_roll = rotation_to_zyx_euler(FR_foot_tip[0:3, 0:3])
        RL_foot_yaw, RL_foot_pitch, RL_foot_roll = rotation_to_zyx_euler(RL_foot_tip[0:3, 0:3])
        RR_foot_yaw, RR_foot_pitch, RR_foot_roll = rotation_to_zyx_euler(RR_foot_tip[0:3, 0:3])

        # To make it clear, three additional frames are attached.
        # Roll frame is attached on rotation frame BEFORE roll rotation is performed.
        # Pitch frame is attached on rotation frame BEFORE pitch rotation is performed.
        # Yaw frame is attached on rotation frame BEFORE yaw rotation is performed, thus
        # it aligns with the world frame axes (though origin is not at the same position).
        FL_transforms[-1] = FL_transforms[-1] @ rotate_x(-FL_foot_roll)
        FL_transforms += [rotate_y(-FL_foot_pitch), rotate_z(-FL_foot_yaw)]
        FL_rotations_axes = self.rotation_axes[0:3] + [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        FR_transforms[-1] = FR_transforms[-1] @ rotate_x(-FR_foot_roll)
        FR_transforms += [rotate_y(-FR_foot_pitch), rotate_z(-FR_foot_yaw)]
        FR_rotations_axes = self.rotation_axes[3:6] + [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        RL_transforms[-1] = RL_transforms[-1] @ rotate_x(-RL_foot_roll)
        RL_transforms += [rotate_y(-RL_foot_pitch), rotate_z(-RL_foot_yaw)]
        RL_rotations_axes = self.rotation_axes[6:9] + [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        RR_transforms[-1] = RR_transforms[-1] @ rotate_x(-RR_foot_roll)
        RR_transforms += [rotate_y(-RR_foot_pitch), rotate_z(-RR_foot_yaw)]
        RR_rotations_axes = self.rotation_axes[9:12] + [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        J_FL = self.get_body_jacobian(body_frame['transform'], FL_transforms, FL_rotations_axes)
        J_FR = self.get_body_jacobian(body_frame['transform'], FR_transforms, FR_rotations_axes)
        J_RL = self.get_body_jacobian(body_frame['transform'], RL_transforms, RL_rotations_axes)
        J_RR = self.get_body_jacobian(body_frame['transform'], RR_transforms, RR_rotations_axes)

        return J_FL, J_FR, J_RL, J_RR

    def get_actuated_body_jacobian(self):
        '''
        Get body Jacobian from actuated joints to body.
        Since the quadruped only have 6 degrees of freedom with 12 actuated joints and 12 unactuated contact joint,
        we need to choose 6 independent actuated joints. The rest of 18 dependent joints include both actuated and
        unactuated joints. We chose FL1, FL2, FR1, FR2, RL3, RR3 as independent joints (This is determined by experiments).
        So the body Jacobian is thus linear mapping from those 6 independent joints velocities to body frame angular velocities.
        :return: Four nearly identical body Jacobians, each of size 6x6.
        '''
        J_FL, J_FR, J_RL, J_RR = self.get_body_jacobians()

        H = np.zeros((18, 24))
        H[0:6, 0:6] = J_FL
        H[0:6, 6:12] = -J_FR
        H[6:12, 6:12] = J_FR
        H[6:12, 12:18] = -J_RL
        H[12:18, 12:18] = J_RL
        H[12:18, 18:24] = -J_RR

        Haa = np.column_stack((H[:, 0:2], H[:, 6:8], H[:, 14], H[:, 20]))
        Hpp = np.column_stack((H[:, 2:6], H[:, 8:14], H[:, 15:20], H[:, 21:24]))

        # chosen_joint_jacobian_rank = np.linalg.matrix_rank(np.column_stack((J_FL[:, 0:2], J_FR[:, 0:2], J_RL[:, 2], J_RR[:, 2])))\
        # print(f"chosen_joint_jacobian_rank: {chosen_joint_jacobian_rank}")
        # print(f"rank(Hpp): {np.linalg.matrix_rank(Hpp)}")

        G = -np.linalg.pinv(Hpp) @ Haa
        G_FL = G[0:4, :] # Joint FL3, FL4, FL5, FL6
        G_FR = G[4:8, :] # Joint FR3, FR4, FR5, FR6
        G_RL = G[8:13, :] # Joint RL1, RL2, RL4, RL5, RL6
        G_RR = G[13:18, :] # Joint RR1, RR2, RR4, RR5, RR6

        M_1 = np.zeros((6, 6))
        M_1[0:2, 0:2] = np.eye(2)
        M_1[2:6, :] = G_FL
        J_b1 = J_FL @ M_1

        M_2 = np.zeros((6, 6))
        M_2[0:2, 2:4] = np.eye(2)
        M_2[2:6, :] = G_FR
        J_b2 = J_FR @ M_2

        M_3 = np.zeros((6, 6))
        M_3[0:2, :] = G_RL[0:2, :]
        M_3[2, 4] = 1
        M_3[3:6, :] = G_RL[2:5, :]
        J_b3 = J_RL @ M_3

        M_4 = np.zeros((6, 6))
        M_4[0:2, :] = G_RR[0:2, :]
        M_4[2, 5] = 1
        M_4[3:6, :] = G_RR[2:5, :]
        J_b4 = J_RR @ M_4

        return J_b1, J_b2, J_b3, J_b4

    def get_body_jacobian(self, body_transform, transformations, rotation_axes):
        '''
        Get body Jacobian from one of the leg.
        :param body_transform:
        :param transformations:
        :param rotation_axes:
        :return:
        '''
        T_i = body_transform
        J_list = []

        p_body = body_transform[0:3, 3]

        for i in range(len(rotation_axes)):
            # Multiply up to the i-th joint to get world frame->joint_i transform
            T_i = T_i @ transformations[i]

            # Joint i's origin in the world frame:
            P_i = T_i[0:3, 3]

            # Vector from joint i's origin to body frame origin
            P_ib = p_body - P_i

            # 'rotation_axes[i]' is the joint axis in joint i's joint frame.
            # Multiply it by T_i's rotation part to express it in the world frame:
            w = T_i[0:3, 0:3] @ rotation_axes[i]

            J = np.concatenate((w, np.cross(w, P_ib)))

            J_list.append(J)

        return np.column_stack(J_list)
    
    def get_foot_tips_IK(self, fl_des, fr_des, rl_des, rr_des):
        '''
        Compute the foot tip inverse kinematics using the Levenberg-Marquardt method with box constraints.
        :param fl_des: Desired foot tip position of front left leg in R^3.
        :param fr_des: Desired foot tip position of front right leg in R^3.
        :param rl_des: Desired foot tip position of rear left leg in R^3.
        :param rr_des: Desired foot tip position of rear right leg in R^3.
        :return: Estimated joint angles that satisfy the inverse kinematics.
        '''
        joints_list = [[self.joints['FL1'], self.joints['FL2'], self.joints['FL3']],
                       [self.joints['FR1'], self.joints['FR2'], self.joints['FR3']],
                       [self.joints['RL1'], self.joints['RL2'], self.joints['RL3']],
                       [self.joints['RR1'], self.joints['RR2'], self.joints['RR3']]]
        original_joints = self.joints.copy()
        x_des_list = [fl_des, fr_des, rl_des, rr_des]
        names = ['FL', 'FR', 'RL', 'RR']

        theta_res={}

        for i in range(4):
            x_des = x_des_list[i]
            joints = np.array(joints_list[i])

            def Jac(theta):
                self.set_joint_angles({f'{names[i]}1': theta[0], f'{names[i]}2': theta[1], f'{names[i]}3': theta[2]})
                J = self.get_foot_tip_jacobians()[i]
                return J
            
            def FK(theta):
                self.set_joint_angles({f'{names[i]}1': theta[0], f'{names[i]}2': theta[1], f'{names[i]}3': theta[2]})
                T = self.get_foot_tip_transforms()[i]
                return T[0:3, 3]
            
            theta = foot_tip_IK_LM(x_des, joints, self.joint_angles_min, self.joint_angles_max, Jac, FK)
            theta_res[names[i]] = theta

        self.joints = original_joints
                
        return theta_res

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
        # World node
        world = {
            'name': 'world',
            'transform': np.eye(4),
            'parent': None,
            'children': {}
        }

        # Body node
        body = {
            'name': 'body',
            'transform': self.body_transform,  # Root node: identity
            'parent': None,          # No parent
            'children': {}
        }

        world['children']['body'] = body

        # -- FRONT LEFT LEG -- #
        # Shoulder 1 node
        FL_s1 = {
            'name': 'FL1',
            'transform': (
                translate(self.Lf, self.Ls, 0) @
                rotate_x(self.joints['FL1'])
            ),
            'parent': body,  # reference the parent node
            'children': {}
        }
        body['children']['FL1'] = FL_s1

        # Shoulder 2 node
        FL_s2 = {
            'name': 'FL2',
            'transform': (
                translate(0, self.Lss, 0) @
                rotate_y(self.joints['FL2'])
            ),
            'parent': FL_s1,
            'children': {}
        }
        FL_s1['children']['FL2'] = FL_s2

        # Elbow node
        FL3 = {
            'name': 'FL3',
            'transform': (
                translate(0, 0, -self.Lleg) @
                rotate_y(self.joints['FL3'])
            ),
            'parent': FL_s2,
            'children': {}
        }
        FL_s2['children']['FL3'] = FL3

        # Foot node (child of elbow)
        FL_foot = {
            'name': 'FL_foot',
            'transform': translate(0, 0, -self.Lfoot),
            'parent': FL3,
            'children': {}
        }
        FL3['children']['FL_foot'] = FL_foot

        # -- FRONT RIGHT LEG -- #
        FR_s1 = {
            'name': 'FR1',
            'transform': (
                translate(self.Lf, -self.Ls, 0) @
                rotate_x(self.joints['FR1'])
            ),
            'parent': body,
            'children': {}
        }
        body['children']['FR1'] = FR_s1

        FR_s2 = {
            'name': 'FR2',
            'transform': (
                translate(0, -self.Lss, 0) @
                rotate_y(self.joints['FR2'])
            ),
            'parent': FR_s1,
            'children': {}
        }
        FR_s1['children']['FR2'] = FR_s2

        FR3 = {
            'name': 'FR3',
            'transform': (
                translate(0, 0, -self.Lleg) @
                rotate_y(self.joints['FR3'])
            ),
            'parent': FR_s2,
            'children': {}
        }
        FR_s2['children']['FR3'] = FR3

        FR_foot = {
            'name': 'FR_foot',
            'transform': translate(0, 0, -self.Lfoot),
            'parent': FR3,
            'children': {}
        }
        FR3['children']['FR_foot'] = FR_foot

        # -- REAR LEFT LEG -- #
        RL_s1 = {
            'name': 'RL1',
            'transform': (
                translate(-self.Lr, self.Ls, 0) @
                rotate_x(self.joints['RL1'])
            ),
            'parent': body,
            'children': {}
        }
        body['children']['RL1'] = RL_s1

        RL_s2 = {
            'name': 'RL2',
            'transform': (
                translate(0, self.Lss, 0) @
                rotate_y(self.joints['RL2'])
            ),
            'parent': RL_s1,
            'children': {}
        }
        RL_s1['children']['RL2'] = RL_s2

        RL3 = {
            'name': 'RL3',
            'transform': (
                translate(0, 0, -self.Lleg) @
                rotate_y(self.joints['RL3'])
            ),
            'parent': RL_s2,
            'children': {}
        }
        RL_s2['children']['RL3'] = RL3

        RL_foot = {
            'name': 'RL_foot',
            'transform': translate(0, 0, -self.Lfoot),
            'parent': RL3,
            'children': {}
        }
        RL3['children']['RL_foot'] = RL_foot

        # -- REAR RIGHT LEG -- #
        RR_s1 = {
            'name': 'RR1',
            'transform': (
                translate(-self.Lr, -self.Ls, 0) @
                rotate_x(self.joints['RR1'])
            ),
            'parent': body,
            'children': {}
        }
        body['children']['RR1'] = RR_s1

        RR_s2 = {
            'name': 'RR2',
            'transform': (
                translate(0, -self.Lss, 0) @
                rotate_y(self.joints['RR2'])
            ),
            'parent': RR_s1,
            'children': {}
        }
        RR_s1['children']['RR2'] = RR_s2

        RR3 = {
            'name': 'RR3',
            'transform': (
                translate(0, 0, -self.Lleg) @
                rotate_y(self.joints['RR3'])
            ),
            'parent': RR_s2,
            'children': {}
        }
        RR_s2['children']['RR3'] = RR3

        RR_foot = {
            'name': 'RR_foot',
            'transform': translate(0, 0, -self.Lfoot),
            'parent': RR3,
            'children': {}
        }
        RR3['children']['RR_foot'] = RR_foot

        # Return the root of the entire kinematic tree
        return world
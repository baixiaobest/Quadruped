import unittest
from quadruped import Quadruped
import numpy as np
from utility import *

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.Lf = 0.3
        self.Lr = 0.3
        self.Ls = 0.2
        self.Lss = 0.1
        self.Lleg = 0.3
        self.Lfoot = 0.4

    def get_quadruped(self):
        quad = Quadruped(self.Lf, self.Lr, self.Ls, self.Lss, self.Lleg, self.Lfoot)
        return quad

    def test_foot_tip_jacobians(self):
        quad = self.get_quadruped()
        joints = quad.get_joint_angles()
        joints['FL2'] = np.pi/4
        joints['FL3'] = -np.pi/2
        joints['FR2'] = np.pi / 4
        joints['FR3'] = -np.pi / 2
        quad.set_joint_angles(joints)
        J_FL, J_FR, J_RL, J_RR = quad.get_foot_tip_jacobians()

        p_fl1 = np.array([1/np.sqrt(2)*(self.Lfoot-self.Lleg), self.Lss, -1/np.sqrt(2)*(self.Lleg+self.Lfoot)])
        p_fr1 = np.array([1/np.sqrt(2)*(self.Lfoot-self.Lleg), -self.Lss, -1/np.sqrt(2)*(self.Lleg+self.Lfoot)])
        w1 = np.array([1, 0, 0])
        J_fl1 = np.concatenate((w1, np.cross(w1, p_fl1)))
        J_fr1 = np.concatenate((w1, np.cross(w1, p_fr1)))

        p2 = np.array([1 / np.sqrt(2) * (self.Lfoot - self.Lleg), 0, -1 / np.sqrt(2) * (self.Lleg + self.Lfoot)])
        w2 = np.array([0, 1, 0])
        J_2 = np.concatenate((w2, np.cross(w2, p2)))

        p3 = np.array([1 / np.sqrt(2) * self.Lfoot, 0, -1 / np.sqrt(2) * self.Lfoot])
        w3 = np.array([0, 1, 0])
        J_3 = np.concatenate((w3, np.cross(w3, p3)))

        J_FL_analytical = np.column_stack((J_fl1, J_2, J_3))
        J_FR_analytical = np.column_stack((J_fr1, J_2, J_3))

        # print(f"numeric\n {J_FR}")
        # print(f"analytical\n {J_FR_analytical}")

        np.testing.assert_allclose(J_FL_analytical, J_FL, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(J_FR_analytical, J_FR, rtol=1e-3, atol=1e-3)

    def test_body_jacobians(self):
        quad = self.get_quadruped()
        joints = quad.get_joint_angles()
        joints['FL2'] = np.pi/4
        joints['FL3'] = -np.pi/2
        joints['FR2'] = np.pi / 4
        joints['FR3'] = -np.pi / 2
        quad.set_joint_angles(joints)
        quad.set_body_tranform(rotate_z(np.pi/2))

        p1 = np.array([self.Ls, -self.Lf, 0])
        w1 = np.array([0, 1, 0])
        J_1 = np.concatenate((w1, np.cross(w1, p1)))

        p2 = np.array([self.Ls+self.Lss, -self.Lf,  0])
        w2 = np.array([-1, 0, 0])
        J_2 = np.concatenate((w2, np.cross(w2, p2)))

        p3 = np.array([self.Ls + self.Lss, 1/np.sqrt(2) * self.Lleg - self.Lf, 1/np.sqrt(2) * self.Lleg])
        w3 = np.array([-1, 0, 0])
        J_3 = np.concatenate((w3, np.cross(w3, p3)))

        p4 = np.array([self.Ls + self.Lss, -(self.Lf + 1/np.sqrt(2)*(self.Lfoot - self.Lleg)), 1/np.sqrt(2)*(self.Lleg + self.Lfoot)])
        w4 = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
        J_4 = np.concatenate((w4, np.cross(w4, p4)))

        p5 = p4
        w5 = np.array([-1, 0, 0])
        J_5 = np.concatenate((w5, np.cross(w5, p5)))

        p6 = p5
        w6 = np.array([0, 0, 1])
        J_6 = np.concatenate((w6, np.cross(w6, p6)))

        J_FL, J_FR, J_RL, J_RR = quad.get_body_jacobians()

        J_FL_analytical = np.column_stack((J_1, J_2, J_3, J_4, J_5, J_6))

        # print(f"numeric\n {J_FL}")
        # print(f"analytical\n {J_FL_analytical}")

        np.testing.assert_allclose(J_FL_analytical, J_FL, rtol=1e-3, atol=1e-3)

    def test_actuated_body_jacobian(self):
        Lf = 1
        Lr = 1
        Ls = 1
        Lss = 1
        Lleg = 1
        Lfoot = 1
        quad = Quadruped(Lf, Lr, Ls, Lss, Lleg, Lfoot)
        joints = quad.get_joint_angles()
        joints['FL1'] = 0.1
        joints['FL2'] = np.pi / 4
        joints['FL3'] = -np.pi / 2

        joints['FR1'] = -0.1
        joints['FR2'] = np.pi / 4
        joints['FR3'] = -np.pi / 2

        joints['RL1'] = 0.1
        joints['RL2'] = np.pi / 4
        joints['RL3'] = -np.pi / 2

        joints['RR1'] = -0.1
        joints['RR2'] = np.pi / 4
        joints['RR3'] = -np.pi / 2
        quad.set_joint_angles(joints)
        # quad.set_body_tranform(rotate_z(np.pi / 2))

        J1, J2, J3, J4 = quad.get_actuated_body_jacobian()
        print(f"J1: \n {J1}")
        print(f"J2: \n {J2}")
        print(f"J3: \n {J3}")
        print(f"J4: \n {J4}")
        print(f"J1 - J2 norm: {np.linalg.norm(J1 - J2)}")
        print(f"J1 - J3 norm: {np.linalg.norm(J1 - J3)}")
        print(f"J1 - J4 norm: {np.linalg.norm(J1 - J4)}")
        print(f"J2 - J3 norm: {np.linalg.norm(J2 - J3)}")
        print(f"J2 - J4 norm: {np.linalg.norm(J2 - J4)}")
        print(f"J3 - J4 norm: {np.linalg.norm(J3 - J4)}")


if __name__ == '__main__':
    unittest.main()

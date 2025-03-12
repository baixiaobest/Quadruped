import unittest
from quadruped import Quadruped
import numpy as np
from utility import *

class MyTestCase(unittest.TestCase):
    def test_foot_tip_jacobians(self):
        Lf = 0.3
        Lr = 0.3
        Ls = 0.2
        Lss = 0.1
        Lleg = 0.3
        Lfoot = 0.4
        quad = Quadruped( Lf, Lr, Ls, Lss, Lleg, Lfoot)
        joints = quad.get_joint_angles()
        joints['FLS2'] = -np.pi/4
        joints['FLE'] = np.pi/2
        joints['FRS2'] = -np.pi / 4
        joints['FRE'] = np.pi / 2
        quad.set_joint_angles(joints)
        J_FL, J_FR, J_RL, J_RR = quad.get_foot_tip_jacobians()

        p_fl1 = np.array([-Lss, 1/np.sqrt(2)*(Lfoot-Lleg), -1/np.sqrt(2)*(Lleg+Lfoot)])
        p_fr1 = np.array([Lss, 1/np.sqrt(2)*(Lfoot-Lleg), -1/np.sqrt(2)*(Lleg+Lfoot)])
        w1 = np.array([0, 1, 0])
        J_fl1 = np.concatenate((w1, np.cross(w1, p_fl1)))
        J_fr1 = np.concatenate((w1, np.cross(w1, p_fr1)))

        p2 = np.array([0, 1 / np.sqrt(2) * (Lfoot - Lleg), -1 / np.sqrt(2) * (Lleg + Lfoot)])
        w2 = np.array([1, 0, 0])
        J_2 = np.concatenate((w2, np.cross(w2, p2)))

        p3 = np.array([0, 1 / np.sqrt(2) * Lfoot, -1 / np.sqrt(2) * Lfoot])
        w3 = np.array([1, 0, 0])
        J_3 = np.concatenate((w3, np.cross(w3, p3)))

        J_FL_analytical = np.column_stack((J_fl1, J_2, J_3))
        J_FR_analytical = np.column_stack((J_fr1, J_2, J_3))

        # print(f"numeric\n {J_FR}")
        # print(f"analytical\n {J_FR_analytical}")

        np.testing.assert_allclose(J_FL_analytical, J_FL, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(J_FR_analytical, J_FR, rtol=1e-3, atol=1e-3)

    def test_body_jacobians(self):
        Lf = 0.3
        Lr = 0.3
        Ls = 0.2
        Lss = 0.1
        Lleg = 0.3
        Lfoot = 0.4
        quad = Quadruped( Lf, Lr, Ls, Lss, Lleg, Lfoot)
        joints = quad.get_joint_angles()
        joints['FLS2'] = -np.pi/4
        joints['FLE'] = np.pi/2
        joints['FRS2'] = -np.pi / 4
        joints['FRE'] = np.pi / 2
        quad.set_joint_angles(joints)
        quad.set_body_tranform(rotate_z(np.pi/2))

        p1 = np.array([Lf, Ls, 0])
        w1 = np.array([-1, 0, 0])
        J_1 = np.concatenate((w1, np.cross(w1, p1)))

        p2 = np.array([Lf, Ls+Lss, 0])
        w2 = np.array([0, 1, 0])
        J_2 = np.concatenate((w2, np.cross(w2, p2)))

        p3 = np.array([Lf - 1/np.sqrt(2) * Lleg, Ls + Lss, 1/np.sqrt(2) * Lleg])
        w3 = np.array([0, 1, 0])
        J_3 = np.concatenate((w3, np.cross(w3, p3)))

        p4 = np.array([Lf + 1/np.sqrt(2)*(Lfoot - Lleg), Ls + Lss, 1/np.sqrt(2)*(Lleg + Lfoot)])
        w4 = np.array([0, 1, 0])
        J_4 = np.concatenate((w4, np.cross(w4, p4)))

        p5 = p4
        w5 = np.array([-1, 0, 0])
        J_5 = np.concatenate((w5, np.cross(w5, p5)))

        p6 = p5
        w6 = np.array([0, 0, 1])
        J_6 = np.concatenate((w6, np.cross(w6, p6)))

        J_FL, J_FR, J_RL, J_RR = quad.get_body_jacobians()

        J_FL_analytical = np.column_stack((J_1, J_2, J_3, J_4, J_5, J_6))

        print(f"numeric\n {J_FL}")
        print(f"analytical\n {J_FL_analytical}")

        np.testing.assert_allclose(J_FL_analytical, J_FL, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    unittest.main()

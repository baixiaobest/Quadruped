import unittest

from quadruped import Quadruped
import numpy as np

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


if __name__ == '__main__':
    unittest.main()

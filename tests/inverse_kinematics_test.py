import sys
import os

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    def test_foot_tip_IK(self):
        quad = self.get_quadruped()
        fl_tip_orig = np.array([self.Lf, self.Ls+self.Lss, -self.Lleg-self.Lfoot])
        fl_tip_des = fl_tip_orig + np.array([0.1, 0.15, 0.1])

        fr_tip_orig = np.array([self.Lf, -self.Ls-self.Lss, -self.Lleg-self.Lfoot])
        fr_tip_des = fr_tip_orig + np.array([0.1, 0.15, 0.1])

        rl_tip_orig = np.array([-self.Lr, self.Ls+self.Lss, -self.Lleg-self.Lfoot])
        rl_tip_des = rl_tip_orig + np.array([0.1, 0.15, 0.1])
        
        rr_tip_orig = np.array([-self.Lr, -self.Ls-self.Lss, -self.Lleg-self.Lfoot])
        rr_tip_des = rr_tip_orig + np.array([0.1, 0.15, 0.1])
        
        theta = quad.get_foot_tips_IK(fl_tip_des, fr_tip_des, rl_tip_des, rr_tip_des)
        print(theta)

        quad.set_joint_angles(theta)
        T_fl, T_fr, T_rl, T_rr = quad.get_foot_tip_transforms()
        fl_tip_pos = T_fl[0:3, 3]
        fr_tip_pos = T_fr[0:3, 3]
        rl_tip_pos = T_rl[0:3, 3]
        rr_tip_pos = T_rr[0:3, 3]

        self.assertTrue(np.allclose(fl_tip_des, fl_tip_pos, atol=1e-3))
        self.assertTrue(np.allclose(fr_tip_des, fr_tip_pos, atol=1e-3))
        self.assertTrue(np.allclose(rl_tip_des, rl_tip_pos, atol=1e-3))
        self.assertTrue(np.allclose(rr_tip_des, rr_tip_pos, atol=1e-3))

if __name__ == '__main__':
    unittest.main()
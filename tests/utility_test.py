import unittest
from utility import *

class MyTestCase(unittest.TestCase):
    def test_something(self):
        yaw = 0.1
        pitch = 0.2
        roll = 0.5
        R = rotate_z(yaw)[0:3, 0:3] @ rotate_y(pitch)[0:3, 0:3] @ rotate_x(roll)[0:3, 0:3]
        y, p, r = rotation_to_zyx_euler(R)
        print(f"{y} {p} {r}")
        self.assertAlmostEqual(y, yaw)  # add assertion here
        self.assertAlmostEqual(p, pitch)
        self.assertAlmostEqual(r, roll)


if __name__ == '__main__':
    unittest.main()

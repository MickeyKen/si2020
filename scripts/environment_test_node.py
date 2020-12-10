#!/usr/bin/env python

import time
import numpy as np
from numpy import inf
import os


import rospy

from test_environment import Env1
from sensor_msgs.msg import LaserScan, JointState
import matplotlib.pyplot as plt

is_training = True

if __name__ == '__main__':
    rospy.init_node('envirnoment_test_node')


    env1 = Env1(is_training, "11311")
    state, rel_dis, yaw, diff_angle, diff_distance, done, arrive = env1.reset(-2.0,3.0,90)


    # state, rel_dis, yaw, diff_angle, diff_distance, done, arrive = env1.step()
    print ("Diff angle: ", diff_angle, "Diff distance", diff_distance)

#!/usr/bin/env python3
import os
import rospy
import numpy as np
import math
from math import pi
import random
import quaternion
import time

from std_msgs.msg import Float64, Int32, Float64MultiArray
from geometry_msgs.msg import Twist, Point, Pose, Vector3, Quaternion
from sensor_msgs.msg import LaserScan, JointState
# from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState, SetModelStateRequest

diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..'
                                , 'models', 'person_standing', 'model.sdf')

PAN_LIMIT = math.radians(90)  #2.9670
TILT_MIN_LIMIT = math.radians(90) - math.atan(3.0/0.998)
TILT_MAX_LIMIT = math.radians(90) - math.atan(1.5/0.998)

PAN_STEP = math.radians(15)
TILT_STEP = math.radians(3)

os.environ['ROS_MASTER_URI'] = "http://localhost:11351" + '/'

class Env1():
    def __init__(self, is_training):
        self.position = Pose()
        self.projector_position = Pose()
        self.goal_position = Pose()
        self.goal_projector_position = Pose()
        self.pub_cmd_vel = rospy.Publisher('/ubiquitous_display/cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('/gazebo/model_states', ModelStates, self.getPose)
        # self.sub_jsp = rospy.Subscriber('/ubiquitous_display/joint_states', JointState, self.getJsp)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.pan_pub = rospy.Publisher('/ubiquitous_display/pan_controller/command', Float64, queue_size=10)
        self.tilt_pub = rospy.Publisher('/ubiquitous_display/tilt_controller/command', Float64, queue_size=10)
        self.image_pub = rospy.Publisher('/ubiquitous_display/image', Int32, queue_size=10)
        self.view_pub = rospy.Publisher('/view', Float64MultiArray, queue_size=10)
        self.past_distance = 0.
        self.past_distance_rate = 0.
        self.past_projector_distance = 0.
        self.yaw = 0
        self.rel_theta = 0
        self.diff_angle = 0
        self.pan_ang = 0.
        self.tilt_ang = 0.
        self.v = 0.
        self.ud_x = 0.

        self.ud_spawn = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        if is_training:
            self.threshold_arrive = 0.25
            self.min_threshold_arrive = 1.5
            self.max_threshold_arrive = 3.0
        else:
            self.threshold_arrive = 0.5
            self.min_threshold_arrive = 1.5
            self.max_threshold_arrive = 3.0

    def constrain(self, input, low, high):
        if input < low:
          input = low
        elif input > high:
          input = high
        else:
          input = input

        return input

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.position.x, self.goal_position.position.y - self.position.position.y)
        self.past_distance = goal_distance

        return goal_distance

    def setUDposition(self, x, y):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            srv = SetModelStateRequest()
            srv.model_state.model_name = 'ubiquitous_display'
            srv.model_state.reference_frame = 'world'  # the same with sdf name
            srv.model_state.pose.position.x = x
            srv.model_state.pose.position.y = y
            srv.model_state.pose.orientation.w = 1
            self.ud_spawn(srv)
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")

    def getPose(self, pose):
        self.position = pose.pose[pose.name.index("ubiquitous_display")]
        self.ud_x = pose.pose[pose.name.index("ubiquitous_display")].position.x
        self.v = pose.twist[pose.name.index("ubiquitous_display")].linear.x
        orientation = self.position.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360

        rel_dis_x = round(self.goal_projector_position.position.x - self.position.position.x, 1)
        rel_dis_y = round(self.goal_projector_position.position.y - self.position.position.y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)

        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

        # view_msg = Float64MultiArray()
        # view_msg.data = [self.position.position.x,self.position.position.y,self.projector_position.position.x,self.projector_position.position.y, self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y]
        # self.view_pub.publish(view_msg)
        # view_plot_pose.view(self.position.position.x,self.position.position.y,self.projector_position.position.x,self.projector_position.position.y, self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y)

    def getProjState(self):
        reach = False

        radian = math.radians(self.yaw) + self.pan_ang + math.radians(90)
        distance = 0.998 * math.tan(math.radians(90) - self.tilt_ang)
        self.projector_position.position.x = distance * math.cos(radian) + self.position.position.x
        self.projector_position.position.y = distance * math.sin(radian) + self.position.position.y
        diff = math.hypot(self.goal_projector_position.position.x - self.projector_position.position.x, self.goal_projector_position.position.y - self.projector_position.position.y)
        # print ("now: ", self.projector_position.position.x, self.projector_position.position.y)
        # print ("goal: ", self.goal_projector_position.position.x, self.goal_projector_position.position.y)
        if diff <= self.threshold_arrive:
            # done = True
            reach = True
        return diff, reach

    def getState(self, scan):
        scan_range = []
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.4 + 0.271
        done = False
        arrive = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(30.)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        current_distance = math.hypot(self.goal_projector_position.position.x- self.position.position.x, self.goal_projector_position.position.y - self.position.position.y)
        if current_distance >= self.min_threshold_arrive and current_distance <= self.max_threshold_arrive:
            # done = True
            arrive = True

        return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

    def setReward(self, done, arrive):
        current_distance = math.hypot(self.goal_projector_position.position.x - self.position.position.x, self.goal_projector_position.position.y - self.position.position.y)
        # distance_rate = (self.past_distance - current_distance)

        if current_distance >= 2.25:
            r_c = ((2.25 - current_distance) ** 2) / 4
        else:
            r_c = 1 - (current_distance / 2.25)

        current_projector_distance, reach = self.getProjState()

        if current_projector_distance >= 0.25:
            r_p = ((0.25 - current_projector_distance) ** 2) / 4
        else:
            r_p = 0

        reward = self.constrain(0.5 - (r_c + r_p + abs(self.v)), 0.0, 0.5)
        # reward = 1 - (r_c + r_p + abs(self.v))

        if done:
            reward = -1.
            self.pub_cmd_vel.publish(Twist())

        if arrive and reach:
            reward = 1.
            self.setUDposition(self.position.position.x, 0.0)

            human = False
            while not human:
                rospy.wait_for_service('/gazebo/delete_model')
                try:
                    self.del_model('actor0')
                except rospy.ServiceException, e:
                    print ("Service call failed: %s" % e)

                rospy.wait_for_service('/gazebo/spawn_sdf_model')
                try:
                    goal_urdf = open(goal_model_dir, "r").read()
                    target = SpawnModel
                    target.model_name = 'actor0'  # the same with sdf name
                    target.model_xml = goal_urdf
                    self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y, self.goal_position.orientation = self.cal_actor_pose(2.5)
                    self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
                except (rospy.ServiceException) as e:
                    print("/gazebo/failed to build the target")
                rospy.sleep(0.1)

                data = None
                while data is None:
                    try:
                        data = rospy.wait_for_message('/scan_filtered', LaserScan, timeout=5)
                    except:
                        pass
                human = self.find_human(data)

            # self.goal_distance = self.getGoalDistace()

        return reward, arrive, reach


    def step(self, action, past_action):

        vel_cmd = Twist()
        if action == 0:
            self.pub_cmd_vel.publish(vel_cmd)
        elif action == 1:
            vel_cmd.linear.x = 0.1
            self.pub_cmd_vel.publish(vel_cmd)
        elif action == 2:
            vel_cmd.linear.x = -0.1
            self.pub_cmd_vel.publish(vel_cmd)
        elif action == 3:
            self.pan_ang = self.constrain(self.pan_ang + PAN_STEP, -PAN_LIMIT, PAN_LIMIT)
            self.pan_pub.publish(self.pan_ang)
        elif action == 4:
            self.pan_ang = self.constrain(self.pan_ang - PAN_STEP, -PAN_LIMIT, PAN_LIMIT)
            self.pan_pub.publish(self.pan_ang)
        elif action == 5:
            self.tilt_ang = self.constrain(self.tilt_ang + TILT_STEP, TILT_MIN_LIMIT, TILT_MAX_LIMIT)
            self.tilt_pub.publish(self.tilt_ang)
        elif action == 6:
            self.tilt_ang = self.constrain(self.tilt_ang - TILT_STEP, TILT_MIN_LIMIT, TILT_MAX_LIMIT)
            self.tilt_pub.publish(self.tilt_ang)
        elif action == 7:
            pass
        else:
            print ("Error action is from 0 to 6")

        time.sleep(0.3)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan_filtered', LaserScan, timeout=5)
            except:
                pass


        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 30. for i in state]

        state.append(past_action)
        state.append(self.pan_ang)
        state.append(self.tilt_ang)
        state.append(self.v)

        # state = state + [yaw / 360, rel_theta / 360, diff_angle / 180]
        reward, arrive, reach = self.setReward(done, arrive)

        return np.asarray(state), reward, done, reach, arrive

    def cal_actor_pose(self, distance):
        xp = 0.
        yp = 0.
        rxp = 0.
        ryp = 0.
        rq = Quaternion()
        xp = random.uniform(-3.0, 3.0)
        yp = random.uniform(3.0, 5.0)
        ang = 0
        rxp = xp + (distance * math.sin(math.radians(ang)))
        ryp = yp - (distance * math.cos(math.radians(ang)))
        q = quaternion.from_euler_angles(0,0,math.radians(ang))
        rq.x = q.x
        rq.y = q.y
        rq.z = q.z
        rq.w = q.w
        return xp, yp, rxp, ryp, rq

    def reset(self):

        # Reset the env #
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            self.del_model('actor0')
        except rospy.ServiceException, e:
            print ("Service call failed: %s" % e)

        self.pan_ang = 0.0
        self.tilt_ang = TILT_MIN_LIMIT

        self.pan_pub.publish(self.pan_ang)
        self.tilt_pub.publish(self.tilt_ang)
        self.pub_cmd_vel.publish(Twist())
        self.setUDposition(0.0, 0.0)

        human = False
        while not human:
            rospy.wait_for_service('/gazebo/delete_model')
            try:
                self.del_model('actor0')
            except rospy.ServiceException, e:
                print ("Service call failed: %s" % e)

            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'actor0'  # the same with sdf name
                target.model_xml = goal_urdf
                self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y, self.goal_position.orientation = self.cal_actor_pose(2.5)
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")
            rospy.sleep(0.1)

            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message('/scan_filtered', LaserScan, timeout=5)
                except:
                    pass
            human = self.find_human(data)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan_filtered', LaserScan, timeout=5)
            except:
                pass


        # self.past_distance_rate, reach = self.getProjState()
        # self.goal_distance = self.getGoalDistace()
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 30. for i in state]

        state.append(0)
        state.append(0.0)
        state.append(TILT_MIN_LIMIT)
        state.append(self.v)

        # state = state + [yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state)

    def find_human(self, data):
        step = math.pi / len(data.ranges)
        human_count = 0
        tf_count = 0
        Human = False
        for count in range(3):
            for i, item in enumerate(data.ranges):
                distance = data.ranges[i]
                x = distance * math.cos(step * (i+1)) + self.ud_x
                y = distance * math.sin(step * (i+1))
                if x < 3.4 and x > -3.4 and y < 5.4 and y > 2.6:
                    human_count += 1
            if human_count > 7:
                tf_count += 1
        if tf_count >= 2:
            Human = True
        else:
            print ("no human", tf_count, human_count)

        return Human

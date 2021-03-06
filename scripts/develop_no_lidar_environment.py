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

diagonal_dis = 6.68
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..'
                                , 'models', 'person_standing', 'model.sdf')
EYE_AREA = 45 #degree

PAN_LIMIT = math.radians(90)  #2.9670
TILT_MIN_LIMIT = math.radians(90) - math.atan(3.0/0.998)
TILT_MAX_LIMIT = math.radians(90) - math.atan(1.5/0.998)
VEL_LIMIT = 0.65

PAN_STEP = math.radians(15)
TILT_STEP = math.radians(3)
VEL_STEP = 0.05

INTIMATE_SPACE = 0.5
class Env1():
    def __init__(self, is_training, ROS_MASTER_URI):
        os.environ['ROS_MASTER_URI'] = "http://localhost:" + str(ROS_MASTER_URI) + '/'
        self.position = Pose()
        self.projector_position = Pose()
        self.goal_position = Pose()
        self.goal_projector_position = Pose()

        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('/gazebo/model_states', ModelStates, self.getPose)

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
        self.pan_ang = 0.
        self.tilt_ang = 0.
        self.vx = 0.
        self.vy = 0.
        self.v = 0.
        self.ud_x = 0.
        self.diff_distance = 0.
        self.diff_angle = 0.

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

    def getGoalDistace(self, x_distance, y_distance):
        goal_distance = math.hypot(x_distance, y_distance)

        return goal_distance

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

        self.yaw = 0


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

    def getState(self, fscan,rscan):
        scan_range = []
        yaw = self.yaw
        min_range = 0.8
        done = False
        arrive = False

        rel_dis_x = round(self.goal_projector_position.position.x - self.position.position.x, 1)
        rel_dis_y = round(self.goal_projector_position.position.y - self.position.position.y, 1)
        diff_distance = math.hypot(rel_dis_x, rel_dis_y)
        rel_dis_hu_x = round(self.goal_position.position.x - self.position.position.x, 1)
        rel_dis_hu_y = round(self.goal_position.position.y - self.position.position.y, 1)
        diff_hu_distance = math.hypot(rel_dis_hu_x, rel_dis_hu_y)

        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2.0 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + abs(math.atan(rel_dis_y / rel_dis_x))
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1.0 / 2.0 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3.0 / 2.0 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi


        if rel_dis_hu_x > 0 and rel_dis_hu_y > 0:
            theta2 = math.atan(rel_dis_hu_y / rel_dis_hu_x)
            h_theta = (1.0 / 2.0 * math.pi) - theta2 + math.radians(self.human_yaw)
        elif rel_dis_hu_x > 0 and rel_dis_hu_y < 0:
            theta2 = 2.0 * math.pi + math.atan(rel_dis_hu_y / rel_dis_hu_x)
            h_theta = math.radians(450) - theta2 + math.radians(self.human_yaw)
        elif rel_dis_hu_x < 0 and rel_dis_hu_y < 0:
            theta2 = math.pi + abs(math.atan(rel_dis_hu_y / rel_dis_hu_x))
            h_theta = theta2 - (1.0 / 2.0 * math.pi) - math.radians(self.human_yaw)
        elif rel_dis_hu_x < 0 and rel_dis_hu_y > 0:
            theta2 = math.pi + math.atan(rel_dis_hu_y / rel_dis_hu_x)
            h_theta = theta2 - (1.0 / 2.0 * math.pi) - math.radians(self.human_yaw)
        elif rel_dis_hu_x == 0 and rel_dis_hu_y > 0:
            theta2 = 1.0 / 2.0 * math.pi
            h_theta = math.radians(self.human_yaw)
        elif rel_dis_hu_x == 0 and rel_dis_hu_y < 0:
            theta2 = 3.0 / 2.0 * math.pi
            h_theta = math.pi - abs(math.radians(self.human_yaw))
        elif rel_dis_hu_y == 0 and rel_dis_hu_x > 0:
            theta2 = 0
            h_theta = (1.0 / 2.0 * math.pi) + math.radians(self.human_yaw)
        else:
            theta2 = math.pi
            h_theta = (1.0 / 2.0 * math.pi) - math.radians(self.human_yaw)


        rel_theta = round(math.degrees(theta), 2)
        diff_angle = abs(rel_theta - yaw)

        rel_h_theta = round(math.degrees(h_theta), 2)
        diff_hu_angle = abs(rel_h_theta)
        # print "rel_dis_x: ",rel_dis_x, ", rel_dis_y: ", rel_dis_y, ", rel_theta: ", rel_theta, ", diff_angle: ", diff_angle

        if diff_hu_angle <= 180:
            diff_hu_angle = round(diff_hu_angle, 2)
        else:
            diff_hu_angle = round(360 - diff_hu_angle, 2)

        scan_range.append(fscan.ranges[540])
        scan_range.append(rscan.ranges[540])

        ### front lidars ###
        if fscan.ranges[900] < rscan.ranges[180]:
            scan_range.append(fscan.ranges[900])
        else:
            scan_range.append(rscan.ranges[180])

        ### rear lidars ###
        if fscan.ranges[180] < rscan.ranges[900]:
            scan_range.append(fscan.ranges[180])
        else:
            scan_range.append(rscan.ranges[900])

        if min_range > min(scan_range) > 0 or diff_hu_distance < INTIMATE_SPACE:
            done = True

        # current_distance = math.hypot(self.goal_projector_position.position.x- self.position.position.x, self.goal_projector_position.position.y - self.position.position.y)
        if diff_distance >= self.min_threshold_arrive and diff_distance <= self.max_threshold_arrive and diff_hu_angle < EYE_AREA:
            # done = True
            arrive = True

        # print "diff_distance: ", diff_distance, ", diff_angle: ", diff_angle

        return scan_range, diff_distance, diff_angle, diff_hu_angle, done, arrive

    def setReward(self, done, arrive):

        _, reach = self.getProjState()

        reward = -1

        if done:
            reward = -200
            self.pub_cmd_vel.publish(Twist())

        if arrive and reach and round(self.v, 1) == 0.0:
            reward = 150
            done = True

        else:
            if arrive:
                reward += 0.4
            if reach:
                reward += 0.4
            if round(self.v, 1) == 0.0:
                reward += 0.2

        return reward, arrive, reach, done


    def step(self, action, past_action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        vel_cmd = Twist()
        if action == 0:
            pass
        elif action == 1:
            self.pub_cmd_vel.publish(vel_cmd)
        elif action == 2:
            self.vx = self.constrain(self.vx + VEL_STEP, -VEL_LIMIT, VEL_LIMIT)
            self.v = math.hypot(self.vx, self.vy)
            if not VEL_LIMIT > self.v:
                self.vx = self.constrain(self.vx - VEL_STEP, -VEL_LIMIT, VEL_LIMIT)
            vel_cmd.linear.x = self.vx
            vel_cmd.linear.y = self.vy
            self.pub_cmd_vel.publish(vel_cmd)
        elif action == 3:
            self.vx = self.constrain(self.vx - VEL_STEP, -VEL_LIMIT, VEL_LIMIT)
            self.v = math.hypot(self.vx, self.vy)
            vel_cmd.linear.x = self.vx
            vel_cmd.linear.y = self.vy
            self.pub_cmd_vel.publish(vel_cmd)
        elif action == 4:
            self.vy = self.constrain(self.vy + VEL_STEP, -VEL_LIMIT, VEL_LIMIT)
            self.v = math.hypot(self.vx, self.vy)
            if not VEL_LIMIT > self.v:
                self.vy = self.constrain(self.vy - VEL_STEP, -VEL_LIMIT, VEL_LIMIT)
            vel_cmd.linear.x = self.vx
            vel_cmd.linear.y = self.vy
            self.pub_cmd_vel.publish(vel_cmd)
        elif action == 5:
            self.vy = self.constrain(self.vy - VEL_STEP, -VEL_LIMIT, VEL_LIMIT)
            self.v = math.hypot(self.vx, self.vy)
            vel_cmd.linear.x = self.vx
            vel_cmd.linear.y = self.vy
            self.pub_cmd_vel.publish(vel_cmd)
        elif action == 6:
            self.pan_ang = self.constrain(self.pan_ang + PAN_STEP, -PAN_LIMIT, PAN_LIMIT)
            self.pan_pub.publish(self.pan_ang)
        elif action == 7:
            self.pan_ang = self.constrain(self.pan_ang - PAN_STEP, -PAN_LIMIT, PAN_LIMIT)
            self.pan_pub.publish(self.pan_ang)
        elif action == 8:
            self.tilt_ang = self.constrain(self.tilt_ang + TILT_STEP, TILT_MIN_LIMIT, TILT_MAX_LIMIT)
            self.tilt_pub.publish(self.tilt_ang)
        elif action == 9:
            self.tilt_ang = self.constrain(self.tilt_ang - TILT_STEP, TILT_MIN_LIMIT, TILT_MAX_LIMIT)
            self.tilt_pub.publish(self.tilt_ang)

        else:
            print ("Error action is from 0 to 9 (10 actions)")

        time.sleep(0.2)

        front_data = None
        while front_data is None:
            try:
                front_data = rospy.wait_for_message('/front_laser_scan', LaserScan, timeout=5)
            except:
                pass
        rear_data = None
        while rear_data is None:
            try:
                rear_data = rospy.wait_for_message('/rear_laser_scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, rel_dis, diff_angle, diff_distance, diff_hu_angle, done, arrive = self.getState(front_data,rear_data)
        state = [i / 25. for i in state]

        state.append(self.constrain(self.pan_ang / PAN_LIMIT, -1.0, 1.0))
        state.append(self.constrain(self.tilt_ang / TILT_MAX_LIMIT, -1.0, 1.0))
        state.append(self.constrain(self.v, -1.0, 1.0))

        state.append(diff_angle / 360)
        state.append(self.constrain(diff_distance / diagonal_dis, -1.0, 1.0))
        state.append(diff_hu_angle / 180)

        print state
        reward, arrive, reach, done = self.setReward(done, arrive)

        return np.asarray(state), reward, done, reach, arrive

    def cal_actor_pose(self, distance):
        xp = 0.
        yp = 0.
        rxp = 0.
        ryp = 0.
        rq = Quaternion()
        xp = random.uniform(-2.8, 2.8)
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

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_world service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        self.pan_ang = 0.0
        self.tilt_ang = TILT_MIN_LIMIT

        self.pan_pub.publish(self.pan_ang)
        self.tilt_pub.publish(self.tilt_ang)
        self.pub_cmd_vel.publish(Twist())


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
                self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y, self.goal_position.orientation = self.cal_actor_pose(x,y,yaw,2.5)
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")

            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message('/scan_filtered', LaserScan, timeout=5)
                except:
                    pass
            human = self.find_human(data)

        front_data = None
        while front_data is None:
            try:
                front_data = rospy.wait_for_message('/front_laser_scan', LaserScan, timeout=5)
            except:
                pass
        rear_data = None
        while rear_data is None:
            try:
                rear_data = rospy.wait_for_message('/rear_laser_scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, rel_dis, diff_angle, diff_distance, diff_hu_angle, done, arrive = self.getState(front_data,rear_data)
        state = [i / 25. for i in state]

        state.append(0.0)
        state.append(TILT_MIN_LIMIT / TILT_MAX_LIMIT)
        state.append(self.constrain(self.v, -1.0, 1.0))

        state.append(diff_angle / 360)
        state.append(self.constrain(diff_distance / diagonal_dis, -1.0, 1.0))
        state.append(diff_hu_angle / 180)
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

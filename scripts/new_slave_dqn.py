#! /usr/bin/env python



import os
import multiprocessing as mp

import rospy
import math
import time
import tf

import numpy as np

import deepq

from std_msgs.msg import Bool
from std_msgs.msg import Float64, Int16
from sensor_msgs.msg import LaserScan

from si2020_msgs.msg import Experiment
from si2020_msgs.srv import *
#adjust as much as you need, limited number of physical cores of your cpu
numberOfCpuCore_to_be_used = 2

continue_execution = False


class Worker(mp.Process):
    def __init__(self, someGlobalNumber, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.port = (name*1) + 50
        self.global_number = someGlobalNumber

        self.stepCounter = 0

        if not continue_execution:

            epochs = 3000000
            steps = 200
            updateTargetNetwork = 1000
            explorationRate = 1
            minibatch_size = 64
            learnStart = 64
            learningRate = 0.00025
            discountFactor = 0.99
            memorySize = 1000000
            network_inputs = 541 + 1 + 2 + 1
            network_outputs = 8

            ### number of hiddenLayer ###
            network_structure = [300,21]
            current_epoch = 0

            self.deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
            self.deepQ.initNetworks(network_structure)

        self.updateTargetNetwork = updateTargetNetwork
        self.learnStart = learnStart


    def run(self):

        #to parallelizing
        os.environ['ROS_MASTER_URI'] = "http://localhost:113" + str(self.port) + '/'
        rospy.init_node('parallelSimulationNode_dqn')

        def saveExperience(msg):
            print "receive"
            observation = msg.observation.data
            action = msg.action.data
            reward = msg.reward.data
            newObservation = msg.newObservation.data
            done = msg.done.data
            self.deepQ.addMemory(np.asarray(observation), action, reward, np.asarray(newObservation), done)

            if self.stepCounter >= self.learnStart:
                if self.stepCounter <= self.updateTargetNetwork:
                    history = self.deepQ.learnOnMiniBatch(minibatch_size, False)
                    # print "pass False"
                else :
                    history = self.deepQ.learnOnMiniBatch(minibatch_size, True)

            if self.stepCounter % self.updateTargetNetwork == 0:
                self.deepQ.updateTargetNetwork()
                print ("updating target network")

            self.stepCounter += 1

        def predictAction(req):
            observation = req.observation.data
            explorationRate = req.explorationRate
            print observation, explorationRate

            qValues = self.deepQ.getQValues(np.asarray(observation))

            print qValues
            action = self.deepQ.selectAction(qValues, explorationRate)

            print action
            res = Int16()
            res.data = action

            return PredictCommandResponse(res)

        sub = rospy.Subscriber("/dqn/experience",Experiment,saveExperience)
        s = rospy.Service('/dqn/predict', PredictCommand, predictAction)
        rate = rospy.Rate(10)



        while not rospy.is_shutdown():
            print "loading"
            rate.sleep()







if __name__ == "__main__":
    #initializing some shared values between process
    global_number = mp.Value('i', 0)

    # parallel training
    workers = [Worker(global_number, i) for i in range(numberOfCpuCore_to_be_used)]
    [w.start() for w in workers]
    [w.join() for w in workers]

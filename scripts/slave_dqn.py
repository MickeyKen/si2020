#!/usr/bin/env python

import rospy
import math
import time
import tf

import deepq

from std_msgs.msg import Bool
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan

from si2020_msgs.msg import Experiment
from si2020_msgs.srv import *

continue_execution = False

class Server(Publishsers):
    def __init__(self):
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
        else:
            #Load weights, monitor info and parameter info.
            #ADD TRY CATCH fro this else
            with open(params_json) as outfile:
                d = json.load(outfile)
                epochs = d.get('epochs')
                steps = d.get('steps')
                updateTargetNetwork = d.get('updateTargetNetwork')
                explorationRate = d.get('explorationRate')
                minibatch_size = d.get('minibatch_size')
                learnStart = d.get('learnStart')
                learningRate = d.get('learningRate')
                discountFactor = d.get('discountFactor')
                memorySize = d.get('memorySize')
                network_inputs = d.get('network_inputs')
                network_outputs = d.get('network_outputs')
                network_structure = d.get('network_structure')
                current_epoch = d.get('current_epoch')

            deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
            self.deepQ.initNetworks(network_structure)
            print ("    ***** load file "+ weight_file+" *****")
            self.deepQ.loadWeights(weight_file)

        self.updateTargetNetwork = updateTargetNetwork
        self.learnStart = learnStart

        self.stepCounter = 0

        # Declare server
        self.subscribe = rospy.Subscriber('/dqn/experience', Experiment, self.experiment_callback)
        self.server = rospy.Service('/dqn/predict', PredictCommand, self.action_callback)


    def experiment_callback(self, msg):
        observation = msg.observationself.data
        action = msg.action.data
        reward = msg.reward.data
        newObservation = msg.newObservation.data
        done = msg.done.data
        self.deepQ.addMemory(observation, action, reward, newObservation, done)

        if self.stepCounter >= self.learnStart:
            if self.stepCounter <= self.updateTargetNetwork:
                history = self.deepQ.learnOnMiniBatch(minibatch_size, False)
                # print "pass False"
            else :
                history = self.deepQ.learnOnMiniBatch(minibatch_size, True)

        if self.stepCounter % self.updateTargetNetwork == 0:
            deepQ.updateTargetNetwork()
            print ("updating target network")

        self.stepCounter += 1

    def action_callback(self, req):
        observation = req.observation.data
        explorationRate = req.explorationRate.data

        qValues = self.deepQ.getQValues(observation)
        action = self.deepQ.selectAction(qValues, explorationRate)

        return PredictCommandResponse(action)


if __name__ == '__main__':
    rospy.init_node('dqn_slave_node', anonymous=True)

    server = Server()

    rospy.spin()

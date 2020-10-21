#!/usr/bin/env python

import gym
from gym import wrappers
# import gym_gazebo
import time
import numpy as np
from numpy import inf
# from distutils.dir_util import copy_tree
import os
import json
# import liveplot
import deepq

import rospy

from no_lidar_no_arrive_environment import Env1



out_path = 'leaned_model_testing_output.txt'
is_training = False

continue_execution = True
weight_file = "3200.h5"
params_json  = '3200.json'


if __name__ == '__main__':


    ROS_MASTER_URI = "11311"

    os.environ['ROS_MASTER_URI'] = "http://localhost:" + ROS_MASTER_URI + '/'
    rospy.init_node('train_frame')

    env1 = Env1(is_training, ROS_MASTER_URI)

    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST
        epochs = 3000000
        steps = 150
        updateTargetNetwork = 10000
        explorationRate = 1
        minibatch_size = 64
        learnStart = 64
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 1000000
        network_inputs = 7
        network_outputs = 8

        ### number of hiddenLayer ###
        network_structure = [56,28]
        current_epoch = 0

        deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
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
        deepQ.initNetworks(network_structure)
        print ("    ***** load file "+ weight_file+" *****")
        deepQ.loadWeights(weight_file)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()

    xx = []
    y = []
    y2 = []

    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation1 = env1.reset()
        cumulated_reward1 = 0
        done1 = False
        episode_step = 0
        last_action1 = 0
        service_count1 = 0

        # run until env returns done
        for i in range(150):

            qValues1 = deepQ.getQValues(observation1)
            action1 = deepQ.selectAction(qValues1, explorationRate)
            newObservation1, reward1, done1, arrive1, reach1  = env1.step(action1, last_action1)
            last_action1 = action1
            cumulated_reward1 += reward1
            if highest_reward < cumulated_reward1:
                highest_reward = cumulated_reward1
            observation1 = newObservation1


            episode_step = i + 1

            if reward1 == 150:
                service_count1 += 1
                done1 = True

            if done1 or episode_step == 150:
                done1 = True
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                print ("EP " + str(epoch) + " - " + format(episode_step) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated1 R: " + str(cumulated_reward1) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                if (epoch)%50==0:
                    parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                    parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                    parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                break


            stepCounter += 1

        # plot(xx,y,xx,y2,cumulated_reward)

        filehandle = open(out_path, 'a+')
        filehandle.write(str(epoch) + ',' + str(episode_step) + ',' + str(cumulated_reward1)+ ',' + str(steps) +  ',' + str(service_count1) + "\n")
        filehandle.close()

        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)

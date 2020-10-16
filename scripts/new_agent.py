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

import rospy

from new_environment import Env1


from si2020_msgs.srv import *
from si2020_msgs.msg import Experiment

out_path = 'output.txt'
loss_out_path = 'output_loss.txt'
is_training = True

continue_execution = False
weight_file = "700.h5"
params_json  = '700.json'

os.environ['ROS_MASTER_URI'] = "http://localhost:11311" + '/'

if __name__ == '__main__':
    rospy.init_node('train_frame')

    call_predict = rospy.ServiceProxy('/dqn/predict', PredictCommand)
    experience_pub = rospy.Publisher('/dqn/experience', Experiment, queue_size=10)

    args = sys.argv
    ROS_MASTER_URI = args[1]
    explorationRate = float(args[2])
    print (str(ROS_MASTER_URI) + " , " + str(explorationRate))

    env = Env1(is_training, ROS_MASTER_URI)
    print "declare environment"

    epochs = 3000000
    steps = 200
    current_epoch = 0

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    highest_reward = 0

    start_time = time.time()

    xx = []
    y = []
    y2 = []

    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation = env.reset()
        print "reset"
        cumulated_reward = 0
        done = False
        episode_step = 0
        last_action = 0
        service_count = 0
        loss_sum = 0.0

        # run until env returns done
        for i in range(200):

            req = PredictCommandRequest()
            req.observation.data = observation
            req.explorationRate = explorationRate
            predict = call_predict(req)
            print ("request" + str(ROS_MASTER_URI))
            action = predict.action.data
            print action

            newObservation, reward, done, arrive, reach  = env.step(action, last_action)
            last_action = action

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            msg = Experiment()
            msg.observation.data = observation
            msg.action.data = action
            msg.reward.data = reward
            msg.newObservation.data = newObservation
            msg.done.data = done
            experience_pub.publish(msg)

            observation = newObservation

            episode_step = i + 1

            if reward == 200:
                service_count += 1
                done = True

            if done or episode_step == 200:
                last100Scores[last100ScoresIndex] = episode_step
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP " + str(epoch) + " - " + format(episode_step) + "/" + str(steps) + " Episode steps   Exploration=" + str(round(explorationRate, 2)))
                else :
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP " + str(epoch) + " - " + format(episode_step) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated1 R: " + str(cumulated_reward1) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                    if (epoch)%50==0:
                        print ("save model")
                        deepQ.saveModel(str(epoch)+'.h5')
                        parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                        parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(str(epoch)+'.json', 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)
                break

        filehandle = open(out_path, 'a+')
        filehandle.write(str(epoch) + ',' + str(episode_step) + ',' + str(cumulated_reward1)+ ',' + str(steps) +  ',' + str(service_count1) + "\n")
        filehandle.close()
        filehandle = open(loss_out_path, 'a+')
        filehandle.write(str(epoch) + ',' + str(loss_sum/float(episode_step)) + "\n")
        filehandle.close()


        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)

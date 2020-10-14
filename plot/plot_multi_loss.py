#!/usr/bin/python
import matplotlib.pyplot as plt

path = 'output_loss.txt'
isServiceCount = False

if __name__ == '__main__':


    xp = []
    yp = []
    average_xp = []
    average_yp = []

    last_step = 0

    sum = 0
    count = 0

    fig = plt.figure()

    plt.ion()
    # plt.title('Simple Curve Graph')
    plt.xlabel('Episode')
    plt.ylabel('loss')
    plt.grid()
    with open(path) as f:
        for s_line in f:
            step = int(s_line.split(',')[0])
            data = s_line.split(',')[1]
            data = float(data)
            xp.append(step)
            yp.append(data)


        plt.plot(xp,yp)
        # plt.plot(average_xp,average_yp, color="#00529a")
        plt.draw()
        fig.savefig("loss_multi.png")
        plt.pause(0)

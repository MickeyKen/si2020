#!/usr/bin/python
import matplotlib.pyplot as plt

path = 'environment_output_test_1223_3.txt'
isServiceCount = True

if __name__ == '__main__':


    xp1 = []
    yp1 = []
    xp2 = []
    yp2 = []
    xp3 = []
    yp3 = []
    xp4 = []
    yp4 = []

    fig = plt.figure()

    plt.ion()
    # plt.title('Simple Curve Graph')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.xlim(0,1500)
    # plt.ylim(-500,2000)
    plt.grid()
    with open(path) as f:

        for s_line in f:
            done_arrive = s_line.split(',')[0]
            if done_arrive == "done":
                xp1.append(float(s_line.split(',')[1]))
                yp1.append(float(s_line.split(',')[2]))
                xp2.append(float(s_line.split(',')[3]))
                yp2.append(float(s_line.split(',')[4]))
            elif done_arrive == "arrive":
                xp3.append(float(s_line.split(',')[1]))
                yp3.append(float(s_line.split(',')[2]))
                xp4.append(float(s_line.split(',')[3]))
                yp4.append(float(s_line.split(',')[4]))

        plt.plot(xp1,yp1, color="b", alpha=0.5, marker="*", linestyle='None')
        # plt.plot(xp2,yp2, color="b", alpha=0.5, marker="o", linestyle='None')
        plt.plot(xp3,yp3, color="r", alpha=0.5, marker="*", linestyle='None')
        # plt.plot(xp4,yp4, color="r", alpha=0.5, marker="o", linestyle='None')
        plt.draw()
        fig.savefig("plot_position_" + str(path.split('.')[0]) + ".png")
        plt.pause(0)

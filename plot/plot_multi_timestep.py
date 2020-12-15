#!/usr/bin/python
import matplotlib.pyplot as plt

path = 'env_max_200_output_test_1213_2.txt'
isServiceCount = True

if __name__ == '__main__':


    xp = []
    yp = []
    yp2 = []
    yp3 = []
    yp4 = []
    average_xp = []
    average_yp = []
    average_yp2 = []
    average_yp3 = []
    average_yp4 = []

    ave_num = 100

    flag = 0
    sum1 = 0
    sum2 = 0

    sum_human = 0
    sum_step = 0

    count = 0

    fig = plt.figure()

    plt.ion()
    # plt.title('Simple Curve Graph')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    # plt.xlim(0,1500)
    # plt.ylim(-500,2000)
    plt.grid()
    with open(path) as f:
        xp.append(0)
        yp.append(0)
        yp2.append(0)
        yp3.append(0)
        for s_line in f:
            step = int(s_line.split(',')[1])
            counter = int(s_line.split(',')[4])

            xp.append(count + 1)
            yp.append(counter)
            if step == 150:
                yp2.append(1) ### 150
                yp3.append(0) ### collisiion
            else:
                yp2.append(0) ### 150
                yp3.append(1) ### collisiion

            num10 = count // ave_num

            if num10 == flag:
                sum1 += float(step)
                sum_human += 1
                sum_step += 1
            else:
                sum1 = sum1 / ave_num
                average_xp.append((flag+1)*ave_num)
                average_yp.append(sum1)
                average_yp2.append(sum_human)
                average_yp3.append(sum_step)
                sum1 = 0
                sum_human = 0
                sum_step = 0
                sum1 += float(step)
                flag += 1
            count += 1

        plt.plot(xp,yp, color="#a9ceec", alpha=0.5)
        print len(yp), len(yp2)
        plt.plot(xp,yp2, color="red", alpha=0.5)
        plt.plot(xp,yp3, color="green", alpha=0.5)
        plt.plot(average_xp,average_yp, color="#00529a")
        plt.draw()
        fig.savefig("result_multi_timestep_" + str(path.split(',')[0]) + ".png")
        plt.pause(0)

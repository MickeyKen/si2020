#!/usr/bin/python
import matplotlib.pyplot as plt

path = 'output.txt'
isServiceCount = True

if __name__ == '__main__':


    xp = []
    yp = []
    yp2 = []
    average_xp = []
    average_yp = []
    average_yp2 = []

    ave_num = 10

    flag = 0
    sum1 = 0
    sum2 = 0

    count = 0

    fig = plt.figure()

    plt.ion()
    # plt.title('Simple Curve Graph')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.xlim(0,1500)
    # plt.ylim(-500,2000)
    plt.grid()
    with open(path) as f:
        xp.append(0)
        yp.append(0)
        average_xp.append(0)
        average_yp.append(0)
        for s_line in f:
            agent = s_line.split(',')[2]
            # print(int(moji.split('.')[0]))
            xp.append(count + 1)
            yp.append(float(agent))

            num10 = count // ave_num

            if num10 == flag:
                sum1 += float(agent)
            else:
                sum1 = sum1 / ave_num
                average_xp.append((flag+1)*ave_num)
                average_yp.append(sum1)
                sum1 = 0
                sum1 += float(agent)
                flag += 1
            count += 1

        plt.plot(xp,yp, color="#a9ceec", alpha=0.5)
        plt.plot(average_xp,average_yp, color="#00529a")
        plt.draw()
        fig.savefig("result_multi_reward_1.png")
        plt.pause(0)

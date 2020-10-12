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
        yp2.append(0)
        average_xp.append(0)
        average_yp.append(0)
        average_yp2.append(0)
        for s_line in f:
            agent1 = s_line.split(',')[2]
            agent2 = s_line.split(',')[3]
            # print(int(moji.split('.')[0]))
            xp.append(count + 1)
            yp.append(int(agent1.split('.')[0]))
            yp2.append(int(agent2.split('.')[0]))

            num10 = count // ave_num

            if num10 == flag:
                sum1 += int(agent1.split('.')[0])
                sum2 += int(agent2.split('.')[0])
            else:
                sum1 = sum1 / ave_num
                sum2 = sum2 / ave_num
                average_xp.append((flag+1)*ave_num)
                average_yp.append(sum1)
                average_yp2.append(sum2)
                sum1 = 0
                sum2 = 0
                sum1 += int(agent1.split('.')[0])
                sum2 += int(agent2.split('.')[0])
                flag += 1
            count += 1

        plt.plot(xp,yp, color="#a9ceec", alpha=0.5)
        plt.plot(xp,yp2, color="#f781bf", alpha=0.5)
        plt.plot(average_xp,average_yp, color="#00529a")
        plt.plot(average_xp,average_yp2, color="#e41a1c")
        plt.draw()
        fig.savefig("result_multi_reward.png")
        plt.pause(0)

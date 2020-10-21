#!/usr/bin/python
import matplotlib.pyplot as plt

path = 'env_max_200_output_loss_test_1019_2.txt'
isServiceCount = False

if __name__ == '__main__':


    xp = []
    yp = []
    average_xp = []
    average_yp = []

    ave_num = 100

    last_step = 0

    sum = 0
    count = 0
    flag = 0

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

            if (count // ave_num) == flag:
                sum += data
            else:
                sum = sum / float(ave_num)
                average_xp.append((flag+1)*ave_num)
                # print (flag+1)*ave_num
                average_yp.append(sum)
                sum = 0
                sum += data
                flag += 1
            count += 1

        plt.plot(xp,yp, alpha=0.5, label="loss per episode")
        plt.plot(average_xp,average_yp, color="#00529a", label="average loss over the 100 last episodes")
        plt.legend( loc='upper left', borderaxespad=1)
        plt.draw()
        fig.savefig("loss_multi.png")
        plt.pause(0)

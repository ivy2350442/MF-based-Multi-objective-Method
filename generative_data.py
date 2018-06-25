import numpy as np
import random
import csv


def main():
    data = []
    #決定生成使用者數量
    user = 2000

    facebook_h = []
    facebook_l = []
    for i in range(user):
        facebook_h.append(random.uniform(0.4, 0.6))
        facebook_l.append(random.uniform(0.2, 0.4))

    google_h = []
    google_l = []
    for i in range(user):
        google_h.append(random.uniform(0.3, 0.5))
        google_l.append(random.uniform(0.1, 0.3))

    youtube_h = []
    youtube_l = []
    for i in range(user):
        youtube_h.append(random.uniform(0.15, 0.3))
        youtube_l.append(random.uniform(0.01, 0.15))


    n1 = []
    n2 = []
    n3 = []
    n4 = []
    n5 = []
    n6 = []
    n7 = []
    n8 = []
    n9 = []
    n10 = []
    n11 = []
    n12 = []
    for i in range(user):
        n1.append(random.uniform(0.5, 0.6))
        n2.append(random.uniform(0.2, 0.5))
        n3.append(random.uniform(0.2, 0.3))
        n4.append(random.uniform(0.15, 0.25 ))
        n5.append(random.uniform(0.65, 0.8))
        n6.append(random.uniform(0.55, 0.6))
        n7.append(random.uniform(0.2, 0.35))
        n8.append(random.uniform(0.01, 0.1))
        n9.append(random.uniform(0.35, 0.45))
        n10.append(random.uniform(0.25, 0.3))
        n11.append(random.uniform(0.3, 0.4))
        n12.append(random.uniform(0.2, 0.4))


    #user id
    for i in range(user):
        data.append([i])

    #user gender
    for i in range(user):
        if i+1 <= 0.45 * user:
            data[i].append(0)
        else:
            data[i].append(1)

    #age
    for i in range(user):
        if i+1 <= 0.045 * user:
            data[i].append(0)
        elif i+1 <= 0.2025 * user:
            data[i].append(1)
        elif i+1 <= 0.3375 * user:
            data[i].append(2)
        elif i+1 <= 0.45 * user:
            data[i].append(3)
        elif i+1 <= 0.5325 * user:
            data[i].append(0)
        elif i+1 <= 0.725 * user:
            data[i].append(1)
        elif i+1 <= 0.89 * user:
            data[i].append(2)
        else:
            data[i].append(3)

    #marr
    for i in range(user):
        #men
        if i+1 <= 0.0315 * user:
            data[i].append(0)
        elif i+1 <= 0.04455 * user:
            data[i].append(1)
        elif i+1 <= 0.045 * user:
            data[i].append(2)
        elif i+1 <= 0.100125 * user:
            data[i].append(0)
        elif i+1 <= 0.178875 * user:
            data[i].append(1)
        elif i+1 <= 0.2025 * user:
            data[i].append(2)
        elif i+1 <= 0.243 * user:
            data[i].append(0)
        elif i+1 <= 0.27 * user:
            data[i].append(1)
        elif i+1 <= 0.3375 * user:
            data[i].append(2)
        elif i+1 <= 0.37125 * user:
            data[i].append(0)
        elif i+1 <= 0.376875 * user:
            data[i].append(1)
        elif i+1 <= 0.45 * user:
            data[i].append(2)
        #women
        elif i+1 <= 0.5094 * user:
            data[i].append(0)
        elif i+1 <= 0.531675 * user:
            data[i].append(1)
        elif i+1 <= 0.5325 * user:
            data[i].append(2)
        elif i+1 <= 0.60565 * user:
            data[i].append(0)
        elif i+1 <= 0.6865 * user:
            data[i].append(1)
        elif i+1 <= 0.725 * user:
            data[i].append(2)
        elif i+1 <= 0.76625 * user:
            data[i].append(0)
        elif i+1 <= 0.791 * user:
            data[i].append(1)
        elif i+1 <= 0.89 * user:
            data[i].append(2)
        elif i+1 <= 0.912 * user:
            data[i].append(0)
        elif i+1 <= 0.9208 * user:
            data[i].append(1)
        else:
            data[i].append(2)

    u = 0
    d = 0
    #facebook
    for i in range(user):
        #men
        if i+1 <= 0.004725 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.0315 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.033066 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.04455 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.04464 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.045 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.058781 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.100125 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.12375 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.178875 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.18549 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.2025 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.2187 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.243 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.25515 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.27 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.29835 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.3375 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.359438 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.37125 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.375469 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.376875 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.435375 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.45 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.45594 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.5094 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.511182 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.531675 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.531799 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.5325 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.54713 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.60565 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.62748 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.6865 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.69805 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.725 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.7415 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.76625 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.776645 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.791 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.82565 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.89 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.9076 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.912 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.9186 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 <= 0.9208 * user:
            data[i].append(facebook_h[u])
            u = u + 1
        elif i+1 <= 0.97624 * user:
            data[i].append(facebook_l[d])
            d = d + 1
        elif i+1 < 1.1 * user:
            data[i].append(facebook_h[u])
            u = u + 1

    n = 0
    d = 0
    #google
    for i in range(user):
        #men
        if i+1 <= 0.036 * user:
            data[i].append(google_l[d])
            d = d + 1
        elif i+1 <= 0.045 * user:
            data[i].append(google_h[n])
            n = n + 1
        elif i+1 <= 0.09225 * user:
            data[i].append(google_l[d])
            d = d + 1
        elif i+1 <= 0.2025 * user:
            data[i].append(google_h[n])
            n = n + 1
        elif i+1 <= 0.24975 * user:
            data[i].append(google_l[d])
            d = d + 1
        elif i+1 <= 0.3375 * user:
            data[i].append(google_h[n])
            n = n + 1
        elif i+1 <= 0.388125 * user:
            data[i].append(google_l[d])
            d = d + 1
        elif i+1 <= 0.45 * user:
            data[i].append(google_h[n])
            n = n + 1
        #women
        elif i+1 <= 0.516 * user:
            data[i].append(google_l[d])
            d = d + 1
        elif i+1 <= 0.5325 * user:
            data[i].append(google_h[n])
            n = n + 1
        elif i+1 <= 0.59025 * user:
            data[i].append(google_l[d])
            d = d + 1
        elif i+1 <= 0.725 * user:
            data[i].append(google_h[n])
            n = n + 1
        elif i+1 <= 0.78275 * user:
            data[i].append(google_l[d])
            d = d + 1
        elif i+1 <= 0.89 * user:
            data[i].append(google_h[n])
            n = n + 1
        elif i+1 <= 0.9395 * user:
            data[i].append(google_l[d])
            d = d + 1
        elif i+1 < 1.1 * user:
            data[i].append(google_h[n])
            n = n + 1

    n = 0
    d = 0
    #Email
    for i in range(user):
        #men
        if i+1 <= 0.0405 * user:
            data[i].append(youtube_l[d])
            d = d + 1
        elif i+1 <= 0.045 * user:
            data[i].append(youtube_h[n])
            n = n + 1
        elif i+1 <= 0.115875 * user:
            data[i].append(youtube_l[d])
            d = d + 1
        elif i+1 <= 0.2025 * user:
            data[i].append(youtube_h[n])
            n = n + 1
        elif i+1 <= 0.2565 * user:
            data[i].append(youtube_l[d])
            d = d + 1
        elif i+1 <= 0.3375 * user:
            data[i].append(youtube_h[n])
            n = n + 1
        elif i+1 <= 0.38025 * user:
            data[i].append(youtube_l[d])
            d = d + 1
        elif i+1 <= 0.45 * user:
            data[i].append(youtube_h[n])
            n = n + 1
        #women
        elif i+1 <= 0.52425 * user:
            data[i].append(youtube_l[d])
            d = d + 1
        elif i+1 <= 0.5325 * user:
            data[i].append(youtube_h[n])
            n = n + 1
        elif i+1 <= 0.619125 * user:
            data[i].append(youtube_l[d])
            d = d + 1
        elif i+1 <= 0.725 * user:
            data[i].append(youtube_h[n])
            n = n + 1
        elif i+1 <= 0.791 * user:
            data[i].append(youtube_l[d])
            d = d + 1
        elif i+1 <= 0.89 * user:
            data[i].append(youtube_h[n])
            n = n + 1
        elif i+1 <= 0.9318 * user:
            data[i].append(youtube_l[d])
            d = d + 1
        elif i+1 < 1.1 * user:
            data[i].append(youtube_h[n])
            n = n + 1


    for i in range(user):
        data[i].append(n1[i])

    for i in range(user):
        data[i].append(n2[i])

    for i in range(user):
        data[i].append(n3[i])

    for i in range(user):
        data[i].append(n4[i])

    for i in range(user):
        data[i].append(n5[i])

    for i in range(user):
        data[i].append(n6[i])

    for i in range(user):
        data[i].append(n7[i])

    for i in range(user):
        data[i].append(n8[i])

    for i in range(user):
        data[i].append(n9[i])

    for i in range(user):
        data[i].append(n10[i])

    for i in range(user):
        data[i].append(n11[i])

    for i in range(user):
        data[i].append(n12[i])


    latent_p = [[0.005, 0.003],[0.001, 0.004],[0.003, 0.002],[0.01, 0.056],[0.04, 0.082],[0.07, 0.064],[0.04, 0.06],[0.03, 0.04],[0.09, 0.01],[0.08, 0.07],[0.02, 0.07],[0.01, 0.04],[0.06, 0.03],[0.07, 0.04],[0.04, 0.05],[0.09, 0.03],[0.08, 0.06],[0.02, 0.07]]

    latent_q = [[0.030,0.064,0.015,0.056,0.045,0.049,0.015,0.035,0.018,0.025,0.074,0.018,0.094,0.024,0.018,0.048,0.035,0.049],[0.085,0.011,0.037,0.045,0.089,0.045,0.078,0.098,0.035,0.085,0.013,0.080,0.018,0.02,0.04,0.06,0.07,0.03]]


    w = np.matmul(latent_p, latent_q)
    print(len(data[0]))
    feature = []
    for i in range(len(data)):
        feature.append([data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7], data[i][8], data[i][9], data[i][10], data[i][11], data[i][12], data[i][13], data[i][14], data[i][15], data[i][16], data[i][17], data[i][18]])


    label_count = np.matmul(feature, w)


    label1 = []
    label2 = []
    label3 = []
    label4 = []
    label5 = []
    label6 = []
    label7 = []
    label8 = []
    label9 = []
    label10 = []
    label11 = []
    label12 = []
    #各種預測項目分別取出
    for i in range(len(label_count)):
        label1.append(label_count[i][0])
        label2.append(label_count[i][1])
        label3.append(label_count[i][2])
        label4.append(label_count[i][3])
        label6.append(label_count[i][5])
        label5.append(label_count[i][4])
        label7.append(label_count[i][6])
        label8.append(label_count[i][7])
        label9.append(label_count[i][8])
        label10.append(label_count[i][9])
        label11.append(label_count[i][10])
        label12.append(label_count[i][11])

    '''
    #印出各項目分界值
    print(np.mean(label1))
    print(np.mean(label2))
    print(np.mean(label3))
    print(np.mean(label4))
    print(np.mean(label5))
    print(np.mean(label6))
    print(np.mean(label7))
    print(np.mean(label8))
    print(np.mean(label9))
    print(np.mean(label10))
    print(np.mean(label11))
    print(np.mean(label12))
    '''

    #用平均判斷上升或下降
    target = []
    for i in range(len(label_count)):
        temp = []


        if label_count[i][0] > np.mean(label1):
            temp.append(10)
        else:
            temp.append(-10)

        if label_count[i][1] > np.mean(label2):
            temp.append(10)
        else:
            temp.append(-10)

        if label_count[i][2] > np.mean(label3):
            temp.append(10)
        else:
            temp.append(-10)

        if label_count[i][3] > np.mean(label4):
            temp.append(10)
        else:
            temp.append(-10)

        if label_count[i][4] > np.mean(label5):
            temp.append(10)
        else:
            temp.append(-10)

        if label_count[i][5] > np.mean(label6):
            temp.append(10)
        else:
            temp.append(-10)

        if label_count[i][6] > np.mean(label7):
            temp.append(10)
        else:
            temp.append(-10)

        if label_count[i][7] > np.mean(label8):
            temp.append(10)
        else:
            temp.append(-10)

        if label_count[i][8] > np.mean(label9):
            temp.append(10)
        else:
            temp.append(-10)

        if label_count[i][9] > np.mean(label10):
            temp.append(10)
        else:
            temp.append(-10)

        if label_count[i][10] > np.mean(label11):
            temp.append(10)
        else:
            temp.append(-10)

        if label_count[i][11] > np.mean(label12):
            temp.append(10)
        else:
            temp.append(-10)


        target.append([data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],temp[0],temp[1],temp[2],temp[3],temp[4],temp[5],temp[6],temp[7],temp[8],temp[9],temp[10],temp[11]])



    #打散資料集
    random.seed(2)
    tmp = list(zip(data, target))
    random.shuffle(tmp)
    data, target = zip(*tmp)


    with open('generative_data_v20001.csv', 'w', newline='') as fout:
        wr = csv.writer(fout)
        title = ["id", "gender", "age", "marr", 'www.facebook.com', 'www.google.com.tw', 'www.youtube.com', 'target1', 'target2', 'target3', 'target4', 'target5', 'target6', 'target7', 'target8', 'target9', 'target10', 'target11', 'target12']
        wr.writerow(title)

        for i in range(user):
            wr.writerow(data[i])
            wr.writerow(target[i])


if __name__ == '__main__':
    main()

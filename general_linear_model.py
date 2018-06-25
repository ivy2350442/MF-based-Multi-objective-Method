#-*- coding: utf-8 -*-
#-*- coding: cp950 -*-

import pandas as pd
import random
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import math
from sklearn.metrics import mean_squared_error

def load_generative_csv():
    data_out_x = []
    data_out_y = []
    title = ["gender", "age", "marr", 'www.facebook.com', 'www.google.com.tw', 'www.youtube.com', 'target1', 'target2', 'target3', 'target4', 'target5', 'target6', 'target7', 'target8', 'target9', 'target10', 'target11', 'target12']

    csv = pd.read_csv("generative_data_v20001.csv")
    for i in range(0, len(csv), 2):
        tmp_x = []
        tmp_y = []
        for cate in title:
            tmp_x.append(csv[cate][i])
            tmp_y.append(csv[cate][i+1])
        data_out_x.append(tmp_x)
        data_out_y.append(tmp_y)

    return data_out_x, data_out_y

def avg_count(y, x):

    label = []
    for i in range(len(x)):
        will = []
        for j in range(6, len(x[0]), 1):
            if x[i][j] - y[i][j] > 0:
                will.append(1)
            if x[i][j] - y[i][j] <= 0:
                will.append(0)
        label.append(will)

    return label

def value2label(true, tr_pred, te_pred):
    one_count = 0
    for value in true:
        if value == 1:
            one_count += 1

    for i in range(one_count):
        if i == one_count - 1:
            line = tr_pred[tr_pred.index(max(tr_pred))]
            tr_pred[tr_pred.index(max(tr_pred))] = -100

        else:
            tr_pred[tr_pred.index(max(tr_pred))] = -100

    for value in range(len(tr_pred)):
        if tr_pred[value] == -100:
            tr_pred[value] = 1

        else:
            tr_pred[value] = 0

    for value in range(len(te_pred)):
        if te_pred[value] >= line:
            te_pred[value] = 1

        else:
            te_pred[value] = 0

    return tr_pred, te_pred

def mfmo(tr_f, tr_t, te_f, te_t):


    #latent_p, latent_q 產生初始值

    latent_w = []
    for i in range(len(tr_f[0])):
        temp = []
        for j in range(len(tr_t[0])):
            temp.append(random.uniform(0.0, 1.0))
        latent_w.append(temp)

    alpha = 0.001 #learning_rate

    #MFMO
    for iters in range(10): #決定學習次數
        for u in range(len(tr_f)):
            for n in range(len(tr_t[0])):
                for k in range(len(tr_f[0])):
                    pre_label = 0.0
                    for l in range(len(tr_f[0])):
                        pre_label = pre_label + (tr_f[u][l] * latent_w[l][n])

                    latent_w[k][n] = -(alpha * 2.0 * (pre_label - tr_t[u][n]) * tr_f[u][k] - latent_w[k][n])



    tr_label = np.matmul(tr_f, latent_w)
    #test 運算
    te_label = np.matmul(te_f, latent_w)

    #把train結果分別拉出來
    tr_s_list = []
    tr_t_list = []
    tr_r_list = []
    tr_ent_list = []
    tr_g_list = []
    tr_e_list = []

    tr_s1_list = []
    tr_t1_list = []
    tr_r1_list = []
    tr_ent1_list = []
    tr_g1_list = []
    tr_e1_list = []

    tr_s_tl = []
    tr_t_tl = []
    tr_r_tl = []
    tr_ent_tl = []
    tr_g_tl = []
    tr_e_tl = []

    tr_s1_tl = []
    tr_t1_tl = []
    tr_r1_tl = []
    tr_ent1_tl = []
    tr_g1_tl = []
    tr_e1_tl = []

    for i in range(len(tr_label)):
        tr_s_list.append(tr_label[i][0])
        tr_t_list.append(tr_label[i][1])
        tr_r_list.append(tr_label[i][2])
        tr_ent_list.append(tr_label[i][3])
        tr_g_list.append(tr_label[i][4])
        tr_e_list.append(tr_label[i][5])

        tr_s1_list.append(tr_label[i][6])
        tr_t1_list.append(tr_label[i][7])
        tr_r1_list.append(tr_label[i][8])
        tr_ent1_list.append(tr_label[i][3])
        tr_g1_list.append(tr_label[i][10])
        tr_e1_list.append(tr_label[i][11])


        tr_s_tl.append(tr_t[i][0])
        tr_t_tl.append(tr_t[i][1])
        tr_r_tl.append(tr_t[i][2])
        tr_ent_tl.append(tr_t[i][3])
        tr_g_tl.append(tr_t[i][4])
        tr_e_tl.append(tr_t[i][5])

        tr_s1_tl.append(tr_t[i][6])
        tr_t1_tl.append(tr_t[i][7])
        tr_r1_tl.append(tr_t[i][8])
        tr_ent1_tl.append(tr_t[i][9])
        tr_g1_tl.append(tr_t[i][10])
        tr_e1_tl.append(tr_t[i][11])


    #把test結果分別拉出來
    te_s_list = []
    te_t_list = []
    te_r_list = []
    te_ent_list = []
    te_g_list = []
    te_e_list = []

    te_s1_list = []
    te_t1_list = []
    te_r1_list = []
    te_ent1_list = []
    te_g1_list = []
    te_e1_list = []

    te_s_tl = []
    te_t_tl = []
    te_r_tl = []
    te_ent_tl = []
    te_g_tl = []
    te_e_tl = []

    te_s1_tl = []
    te_t1_tl = []
    te_r1_tl = []
    te_ent1_tl = []
    te_g1_tl = []
    te_e1_tl = []

    for i in range(len(te_label)):
        te_s_list.append(te_label[i][0])
        te_t_list.append(te_label[i][1])
        te_r_list.append(te_label[i][2])
        te_ent_list.append(te_label[i][3])
        te_g_list.append(te_label[i][4])
        te_e_list.append(te_label[i][5])

        te_s1_list.append(te_label[i][6])
        te_t1_list.append(te_label[i][7])
        te_r1_list.append(te_label[i][8])
        te_ent1_list.append(te_label[i][9])
        te_g1_list.append(te_label[i][10])
        te_e1_list.append(te_label[i][11])


        te_s_tl.append(te_t[i][0])
        te_t_tl.append(te_t[i][1])
        te_r_tl.append(te_t[i][2])
        te_ent_tl.append(te_t[i][3])
        te_g_tl.append(te_t[i][4])
        te_e_tl.append(te_t[i][5])

        te_s1_tl.append(te_t[i][6])
        te_t1_tl.append(te_t[i][7])
        te_r1_tl.append(te_t[i][8])
        te_ent1_tl.append(te_t[i][9])
        te_g1_tl.append(te_t[i][10])
        te_e1_tl.append(te_t[i][11])


    tr_s_list, te_s_list = value2label(tr_s_tl, tr_s_list, te_s_list)
    tr_s_score = mean_squared_error(tr_s_tl, tr_s_list)

    tr_t_list, te_t_list = value2label(tr_t_tl, tr_t_list, te_t_list)
    tr_t_score = mean_squared_error(tr_t_tl, tr_t_list)

    tr_r_list, te_r_list = value2label(tr_r_tl, tr_r_list, te_r_list)
    tr_r_score = mean_squared_error(tr_r_tl, tr_r_list)

    tr_ent_list, te_ent_list = value2label(tr_ent_tl, tr_ent_list, te_ent_list)
    tr_ent_score = mean_squared_error(tr_ent_tl, tr_ent_list)

    tr_g_list, te_g_list = value2label(tr_g_tl, tr_g_list, te_g_list)
    tr_g_score = mean_squared_error(tr_g_tl, tr_g_list)

    tr_e_list, te_e_list = value2label(tr_e_tl, tr_e_list, te_e_list)
    tr_e_score = mean_squared_error(tr_e_tl, tr_e_list)

    tr_s1_list, te_s1_list = value2label(tr_s1_tl, tr_s1_list, te_s1_list)
    tr_s1_score = mean_squared_error(tr_s1_tl, tr_s1_list)

    tr_t1_list, te_t1_list = value2label(tr_t1_tl, tr_t1_list, te_t1_list)
    tr_t1_score = mean_squared_error(tr_t1_tl, tr_t1_list)

    tr_r1_list, te_r1_list = value2label(tr_r1_tl, tr_r1_list, te_r1_list)
    tr_r1_score = mean_squared_error(tr_r1_tl, tr_r1_list)

    tr_ent1_list, te_ent1_list = value2label(tr_ent1_tl, tr_ent1_list, te_ent1_list)
    tr_ent1_score = mean_squared_error(tr_ent1_tl, tr_ent1_list)

    tr_g1_list, te_g1_list = value2label(tr_g1_tl, tr_g1_list, te_g1_list)
    tr_g1_score = mean_squared_error(tr_g1_tl, tr_g1_list)

    tr_e1_list, te_e1_list = value2label(tr_e1_tl, tr_e1_list, te_e1_list)
    tr_e1_score = mean_squared_error(tr_e1_tl, tr_e1_list)

    te_s_score = mean_squared_error(te_s_tl, te_s_list)
    te_t_score = mean_squared_error(te_t_tl, te_t_list)
    te_r_score = mean_squared_error(te_r_tl, te_r_list)
    te_ent_score = mean_squared_error(te_ent_tl, te_ent_list)
    te_g_score = mean_squared_error(te_g_tl, te_g_list)
    te_e_score = mean_squared_error(te_e_tl, te_e_list)

    te_s1_score = mean_squared_error(te_s1_tl, te_s1_list)
    te_t1_score = mean_squared_error(te_t1_tl, te_t1_list)
    te_r1_score = mean_squared_error(te_r1_tl, te_r1_list)
    te_ent1_score = mean_squared_error(te_ent1_tl, te_ent1_list)
    te_g1_score = mean_squared_error(te_g1_tl, te_g1_list)
    te_e1_score = mean_squared_error(te_e1_tl, te_e1_list)

    return tr_s_score, te_s_score, tr_t_score, te_t_score, tr_r_score, te_r_score, tr_ent_score, te_ent_score, tr_g_score, te_g_score, tr_e_score, te_e_score, tr_s1_score, te_s1_score, tr_t1_score, te_t1_score, tr_r1_score, te_r1_score, tr_ent1_score, te_ent1_score, tr_g1_score, te_g1_score, tr_e1_score, te_e1_score

def increase_or_not(label):
    s_count_will = 0
    s_count_not = 0
    t_count_will = 0
    t_count_not = 0
    r_count_will = 0
    r_count_not = 0
    ent_count_will = 0
    ent_count_not = 0
    g_count_will = 0
    g_count_not = 0
    e_count_will = 0
    e_count_not = 0

    s1_count_will = 0
    s1_count_not = 0
    t1_count_will = 0
    t1_count_not = 0
    r1_count_will = 0
    r1_count_not = 0
    ent1_count_will = 0
    ent1_count_not = 0
    g1_count_will = 0
    g1_count_not = 0
    e1_count_will = 0
    e1_count_not = 0


    for i in range(len(label)):
        if label[i][0] == 1:
            s_count_will += 1
        else:
            s_count_not += 1

        if label[i][1] == 1:
            t_count_will += 1
        else:
            t_count_not += 1

        if label[i][2] == 1:
            r_count_will += 1
        else:
            r_count_not += 1

        if label[i][3] == 1:
            ent_count_will += 1
        else:
            ent_count_not += 1

        if label[i][4] == 1:
            g_count_will += 1
        else:
            g_count_not += 1

        if label[i][5] == 1:
            e_count_will += 1
        else:
            e_count_not += 1

        if label[i][6] == 1:
            s1_count_will += 1
        else:
            s1_count_not += 1

        if label[i][7] == 1:
            t1_count_will += 1
        else:
            t1_count_not += 1

        if label[i][8] == 1:
            r1_count_will += 1
        else:
            r1_count_not += 1

        if label[i][9] == 1:
            ent1_count_will += 1
        else:
            ent1_count_not += 1

        if label[i][10] == 1:
            g1_count_will += 1
        else:
            g1_count_not += 1

        if label[i][11] == 1:
            e1_count_will += 1
        else:
            e1_count_not += 1


    print("Shopping_Proportionate increase: " + str(s_count_will))
    print("Shopping_Proportionate decreased: " + str(s_count_not))
    print("Travel_Proportionate increase: " + str(t_count_will))
    print("Travel_Proportionate decreased: " + str(t_count_not))
    print("Restaurant and Dining_Proportionate increase: " + str(r_count_will))
    print("Restaurant and Dining_Proportionate decreased: " + str(r_count_not))
    print("Entertainment_Proportionate increase: " + str(ent_count_will))
    print("Entertainment_Proportionate decreased: " + str(ent_count_not))
    print("Games_Proportionate increase: " + str(g_count_will))
    print("Games_Proportionate decreased: " + str(g_count_not))
    print("Education_Proportionate increase: " + str(e_count_will))
    print("Education_Proportionate decreased: " + str(e_count_not))


    print("Shopping_Proportionate increase: " + str(s1_count_will))
    print("Shopping_Proportionate decreased: " + str(s1_count_not))
    print("Travel_Proportionate increase: " + str(t1_count_will))
    print("Travel_Proportionate decreased: " + str(t1_count_not))
    print("Restaurant and Dining_Proportionate increase: " + str(r1_count_will))
    print("Restaurant and Dining_Proportionate decreased: " + str(r1_count_not))
    print("Entertainment_Proportionate increase: " + str(ent1_count_will))
    print("Entertainment_Proportionate decreased: " + str(ent1_count_not))
    print("Games_Proportionate increase: " + str(g1_count_will))
    print("Games_Proportionate decreased: " + str(g1_count_not))
    print("Education_Proportionate increase: " + str(e1_count_will))
    print("Education_Proportionate decreased: " + str(e1_count_not))


def main():

    data_x = []
    data_y = []

    print("csv loading...")
    data_x, data_y = load_generative_csv()

    label = avg_count(data_x, data_y)
    print(len(label))
    #算有多少上升多少下降
    increase_or_not(label)

    #feature取出來
    feature = []
    for i in range(len(data_x)):
        feature.append([data_x[i][0], data_x[i][1], data_x[i][2], data_x[i][3], data_x[i][4], data_x[i][5], data_x[i][6], data_x[i][7], data_x[i][8], data_x[i][9], data_x[i][10], data_x[i][11], data_x[i][12], data_x[i][13], data_x[i][14], data_x[i][15], data_x[i][16], data_x[i][17]])

    tr_f = []
    tr_t = []

    #決定training data數量
    for i in range(900):
        tr_f.append(feature[i])
        tr_t.append(label[i])

    print(len(tr_f))
    print(len(tr_t))

    te_f = []
    te_t = []
    for i in range(100):
        te_f.append(feature[i+1899])
        te_t.append(feature[i+1899])

    print(len(te_f))
    print(len(te_f))
    #input()

    tr_s_auc = []
    tr_t_auc = []
    tr_r_auc = []
    tr_ent_auc = []
    tr_g_auc = []
    tr_e_auc = []

    tr_s1_auc = []
    tr_t1_auc = []
    tr_r1_auc = []
    tr_ent1_auc = []
    tr_g1_auc = []
    tr_e1_auc = []

    te_s_auc = []
    te_t_auc = []
    te_r_auc = []
    te_ent_auc = []
    te_g_auc = []
    te_e_auc = []

    te_s1_auc = []
    te_t1_auc = []
    te_r1_auc = []
    te_ent1_auc = []
    te_g1_auc = []
    te_e1_auc = []

    tr_s_au, te_s_au, tr_t_au, te_t_au, tr_r_au, te_r_au, tr_ent_au, te_ent_au, tr_g_au, te_g_au, tr_e_au, te_e_au, tr_s1_au, te_s1_au, tr_t1_au, te_t1_au, tr_r1_au, te_r1_au, tr_ent1_au, te_ent1_au, tr_g1_au, te_g1_au, tr_e1_au, te_e1_au = mfmo(tr_f, tr_t, te_f, te_t)


    tr_s_auc.append(tr_s_au)
    tr_t_auc.append(tr_t_au)
    tr_r_auc.append(tr_r_au)
    tr_ent_auc.append(tr_ent_au)
    tr_g_auc.append(tr_g_au)
    tr_e_auc.append(tr_e_au)

    tr_s1_auc.append(tr_s1_au)
    tr_t1_auc.append(tr_t1_au)
    tr_r1_auc.append(tr_r1_au)
    tr_ent1_auc.append(tr_ent1_au)
    tr_g1_auc.append(tr_g1_au)
    tr_e1_auc.append(tr_e1_au)


    te_s_auc.append(te_s_au)
    te_t_auc.append(te_t_au)
    te_r_auc.append(te_r_au)
    te_ent_auc.append(te_ent_au)
    te_g_auc.append(te_g_au)
    te_e_auc.append(te_e_au)

    te_s1_auc.append(te_s1_au)
    te_t1_auc.append(te_t1_au)
    te_r1_auc.append(te_r1_au)
    te_ent1_auc.append(te_ent1_au)
    te_g1_auc.append(te_g1_au)
    te_e1_auc.append(te_e1_au)


    print("mf_training_auc")
    print("label1:  " + str("%.3f" % np.mean(tr_s_auc)))
    print("label2:  " + str("%.3f" % np.mean(tr_t_auc)))
    print("label3:  " + str("%.3f" % np.mean(tr_r_auc)))
    print("label4:  " + str("%.3f" % np.mean(tr_ent_auc)))
    print("label5:  " + str("%.3f" % np.mean(tr_g_auc)))
    print("label6:  " + str("%.3f" % np.mean(tr_e_auc)))
    print("label7:  " + str("%.3f" % np.mean(tr_s1_auc)))
    print("label8:  " + str("%.3f" % np.mean(tr_t1_auc)))
    print("label9:  " + str("%.3f" % np.mean(tr_r1_auc)))
    print("label10:  " + str("%.3f" % np.mean(tr_ent1_auc)))
    print("label11:  " + str("%.3f" % np.mean(tr_g1_auc)))
    print("label12:  " + str("%.3f" % np.mean(tr_e1_auc)))

    print("mf_test_auc")
    print("label1:  " + str("%.3f" % np.mean(te_s_auc)))
    print("label2:  " + str("%.3f" % np.mean(te_t_auc)))
    print("label3:  " + str("%.3f" % np.mean(te_r_auc)))
    print("label4:  " + str("%.3f" % np.mean(te_ent_auc)))
    print("label5:  " + str("%.3f" % np.mean(te_g_auc)))
    print("label6:  " + str("%.3f" % np.mean(te_e_auc)))
    print("label7:  " + str("%.3f" % np.mean(te_s1_auc)))
    print("label8:  " + str("%.3f" % np.mean(te_t1_auc)))
    print("label9:  " + str("%.3f" % np.mean(te_r1_auc)))
    print("label10:  " + str("%.3f" % np.mean(te_ent1_auc)))
    print("label11:  " + str("%.3f" % np.mean(te_g1_auc)))
    print("label12:  " + str("%.3f" % np.mean(te_e1_auc)))


    tr_avg_score = []
    te_avg_score = []
    print('')
    print("mf_training_auc")
    tr_avg_score.append(tr_s_auc)
    tr_avg_score.append(tr_t_auc)
    tr_avg_score.append(tr_r_auc)
    tr_avg_score.append(tr_ent_auc)
    tr_avg_score.append(tr_g_auc)
    tr_avg_score.append(tr_e_auc)
    tr_avg_score.append(tr_s1_auc)
    tr_avg_score.append(tr_t1_auc)
    tr_avg_score.append(tr_r1_auc)
    tr_avg_score.append(tr_ent1_auc)
    tr_avg_score.append(tr_g1_auc)
    tr_avg_score.append(tr_e1_auc)
    print('tr: ' + str(np.mean(tr_avg_score)))


    print('')
    print("mf_test_auc")
    te_avg_score.append(te_s_auc)
    te_avg_score.append(te_t_auc)
    te_avg_score.append(te_r_auc)
    te_avg_score.append(te_ent_auc)
    te_avg_score.append(te_g_auc)
    te_avg_score.append(te_e_auc)
    te_avg_score.append(te_s1_auc)
    te_avg_score.append(te_t1_auc)
    te_avg_score.append(te_r1_auc)
    te_avg_score.append(te_ent1_auc)
    te_avg_score.append(te_g1_auc)
    te_avg_score.append(te_e1_auc)

    print('te: ' + str(np.mean(te_avg_score)))


if __name__ == '__main__':
    main()

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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

def load_generative_csv():
    data_out_x = []
    data_out_y = []
    title = ['gender', 'age', 'marr', 'www.facebook.com', 'www.google.com.tw', 'www.youtube.com', 'Shopping', 'Travel', 'Restaurant and Dining', 'Entertainment', 'Games', 'Education']

    csv = pd.read_csv("user_feature.csv")
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
    #y ~ 12/12
    #x 12/12 ~ 12/25
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

def feature_one_hot(data_y):

    feature_x = []

    for i in range(len(data_y)):
        feature = []

        #gender
        if data_y[i][0] == 0:
            feature.append(1)
            feature.append(0)
            feature.append(0)

        elif data_y[i][0] == 1:
            feature.append(0)
            feature.append(1)
            feature.append(0)

        else:
            feature.append(0)
            feature.append(0)
            feature.append(1)

        #age
        if data_y[i][1] == 0:
            feature.append(1)
            feature.append(0)
            feature.append(0)
            feature.append(0)

        elif data_y[i][1] == 1:
            feature.append(0)
            feature.append(1)
            feature.append(0)
            feature.append(0)

        elif data_y[i][1] == 2:
            feature.append(0)
            feature.append(0)
            feature.append(1)
            feature.append(0)

        else:
            feature.append(0)
            feature.append(0)
            feature.append(0)
            feature.append(1)

        #marr
        if data_y[i][2] == 0:
            feature.append(1)
            feature.append(0)
            feature.append(0)
            feature.append(0)

        elif data_y[i][2] == 1:
            feature.append(0)
            feature.append(1)
            feature.append(0)
            feature.append(0)

        elif data_y[i][2] == 2:
            feature.append(0)
            feature.append(0)
            feature.append(1)
            feature.append(0)

        else:
            feature.append(0)
            feature.append(0)
            feature.append(0)
            feature.append(1)

        feature.append(data_y[i][3])
        feature.append(data_y[i][4])
        feature.append(data_y[i][5])

        feature.append(data_y[i][6])
        feature.append(data_y[i][7])
        feature.append(data_y[i][8])
        feature.append(data_y[i][9])
        feature.append(data_y[i][10])
        feature.append(data_y[i][11])

        feature_x.append(feature)

    return feature_x

def cross_validation(feature, label, run, cv):

    total_item = len(label)
    sitems = math.floor(total_item / cv)
    litems = math.ceil(total_item / cv)
    l = (total_item % cv)
    s = run - (total_item % cv)

    cv_feature = []
    cv_label = []

    #lase_run
    if run >= total_item % cv :
        for item in range(sitems):
            cv_feature.append(feature[l*litems + s*sitems + item])
            cv_label.append(label[l*litems + s*sitems + item])


    else:
        for item in range(litems):
            cv_feature.append(feature[run*litems + item])
            cv_label.append(label[run*litems + item])


    return cv_feature, cv_label

def main():

    data_x = []
    data_y = []

    print("csv loading...")
    data_x, data_y = load_generative_csv()
    #data_x ~ 12/12
    #data_y 12/12 ~ 12/25
    label = avg_count(data_x, data_y)

    increase_or_not(label)

    #feature取出來
    feature = []
    for i in range(len(data_x)):
        feature.append([data_x[i][0], data_x[i][1], data_x[i][2], data_x[i][3], data_x[i][4], data_x[i][5], data_x[i][6], data_x[i][7], data_x[i][8], data_x[i][9], data_x[i][10], data_x[i][11]])

    #打散資料集
    random.seed(2)
    tmp = list(zip(feature, label))
    random.shuffle(tmp)
    feature, label = zip(*tmp)

    feature = feature_one_hot(feature)

    cv = 5
    tr_s_f1 = []
    tr_t_f1 = []
    tr_r_f1 = []
    tr_ent_f1 = []
    tr_g_f1 = []
    tr_e_f1 = []

    te_s_f1 = []
    te_t_f1 = []
    te_r_f1 = []
    te_ent_f1 = []
    te_g_f1 = []
    te_e_f1 = []

    tr_s_auc = []
    tr_t_auc = []
    tr_r_auc = []
    tr_ent_auc = []
    tr_g_auc = []
    tr_e_auc = []

    te_s_auc = []
    te_t_auc = []
    te_r_auc = []
    te_ent_auc = []
    te_g_auc = []
    te_e_auc = []


    for test in range(cv):
        print("------------------ " + str(test+1) + " ------------------")
        tr_feature = []
        tr_label = []
        te_feature = []
        te_label = []

        for cvi in range(cv):
            cv_feature = []
            cv_label = []
            cv_feature, cv_label = cross_validation(feature, label, cvi, cv=cv)

            if cvi == test:
                te_feature = te_feature + cv_feature
                te_label = te_label + cv_label
            else:
                tr_feature = tr_feature + cv_feature
                tr_label = tr_label + cv_label

        #knn
        s_knn = svm.SVC(C = 1.0, kernel='linear', probability=True, random_state=57)
        t_knn = svm.SVC(C = 1.0, kernel='linear', probability=True, random_state=1685)
        r_knn = svm.SVC(C = 2.9, kernel='linear', probability=True, random_state=57)
        ent_knn = svm.SVC(C = 4.8, kernel='linear', probability=True, random_state=57)
        g_knn = svm.SVC(C = 0.1, kernel='linear', probability=True, random_state=57)
        e_knn = svm.SVC(C = 0.05, kernel='linear', probability=True, random_state=27)

        tr_s_tl = []
        tr_t_tl = []
        tr_r_tl = []
        tr_ent_tl = []
        tr_g_tl = []
        tr_e_tl = []

        for i in range(len(tr_label)):
            tr_s_tl.append(tr_label[i][0])
            tr_t_tl.append(tr_label[i][1])
            tr_r_tl.append(tr_label[i][2])
            tr_ent_tl.append(tr_label[i][3])
            tr_g_tl.append(tr_label[i][4])
            tr_e_tl.append(tr_label[i][5])

        te_s_tl = []
        te_t_tl = []
        te_r_tl = []
        te_ent_tl = []
        te_g_tl = []
        te_e_tl = []

        for i in range(len(te_label)):
            te_s_tl.append(te_label[i][0])
            te_t_tl.append(te_label[i][1])
            te_r_tl.append(te_label[i][2])
            te_ent_tl.append(te_label[i][3])
            te_g_tl.append(te_label[i][4])
            te_e_tl.append(te_label[i][5])

        s_knn = s_knn.fit(tr_feature, tr_s_tl)
        t_knn = t_knn.fit(tr_feature, tr_t_tl)
        r_knn = r_knn.fit(tr_feature, tr_r_tl)
        ent_knn = ent_knn.fit(tr_feature, tr_ent_tl)
        g_knn = g_knn.fit(tr_feature, tr_g_tl)
        e_knn = e_knn.fit(tr_feature, tr_e_tl)


        #training預測
        s_knn_tr_pred = s_knn.predict(tr_feature)
        t_knn_tr_pred = t_knn.predict(tr_feature)
        r_knn_tr_pred = r_knn.predict(tr_feature)
        ent_knn_tr_pred = ent_knn.predict(tr_feature)
        g_knn_tr_pred = g_knn.predict(tr_feature)
        e_knn_tr_pred = e_knn.predict(tr_feature)
        #test預測
        s_knn_te_pred = s_knn.predict(te_feature)
        t_knn_te_pred = t_knn.predict(te_feature)
        r_knn_te_pred = r_knn.predict(te_feature)
        ent_knn_te_pred = ent_knn.predict(te_feature)
        g_knn_te_pred = g_knn.predict(te_feature)
        e_knn_te_pred = e_knn.predict(te_feature)



        tr_s_f1.append(f1_score(tr_s_tl, s_knn_tr_pred, average='macro'))
        tr_t_f1.append(f1_score(tr_t_tl, t_knn_tr_pred, average='macro'))
        tr_r_f1.append(f1_score(tr_r_tl, r_knn_tr_pred, average='macro'))
        tr_ent_f1.append(f1_score(tr_ent_tl, ent_knn_tr_pred, average='macro'))
        tr_g_f1.append(f1_score(tr_g_tl, g_knn_tr_pred, average='macro'))
        tr_e_f1.append(f1_score(tr_e_tl, e_knn_tr_pred, average='macro'))



        te_s_f1.append(f1_score(te_s_tl, s_knn_te_pred, average='macro'))
        te_t_f1.append(f1_score(te_t_tl, t_knn_te_pred, average='macro'))
        te_r_f1.append(f1_score(te_r_tl, r_knn_te_pred, average='macro'))
        te_ent_f1.append(f1_score(te_ent_tl, ent_knn_te_pred, average='macro'))
        te_g_f1.append(f1_score(te_g_tl, g_knn_te_pred, average='macro'))
        te_e_f1.append(f1_score(te_e_tl, e_knn_te_pred, average='macro'))


        '''
        s_tmp_tr_y = np.array(tr_s_tl)
        t_tmp_tr_y = np.array(tr_t_tl)
        r_tmp_tr_y = np.array(tr_r_tl)
        ent_tmp_tr_y = np.array(tr_ent_tl)
        g_tmp_tr_y = np.array(tr_g_tl)
        e_tmp_tr_y = np.array(tr_e_tl)

        s_tmp_te_y = np.array(te_s_tl)
        t_tmp_te_y = np.array(te_t_tl)
        r_tmp_te_y = np.array(te_r_tl)
        ent_tmp_te_y = np.array(te_ent_tl)
        g_tmp_te_y = np.array(te_g_tl)
        e_tmp_te_y = np.array(te_e_tl)

        #train auc
        s_knn_tr_prob = s_knn.predict(tr_feature)
        s_knn_tr_pred = np.array(s_knn_tr_prob)
        s_knn_tr_fpr, s_knn_tr_tpr, s_knn_tr_thresholds = roc_curve(s_tmp_tr_y, s_knn_tr_pred)
        s_knn_tr_roc_auc = "%.3f" % auc(s_knn_tr_fpr, s_knn_tr_tpr)
        tr_s_auc.append(float(s_knn_tr_roc_auc))

        t_knn_tr_prob = t_knn.predict(tr_feature)
        t_knn_tr_pred = np.array(t_knn_tr_prob)
        t_knn_tr_fpr, t_knn_tr_tpr, t_knn_tr_thresholds = roc_curve(t_tmp_tr_y, t_knn_tr_pred)
        t_knn_tr_roc_auc = "%.3f" % auc(t_knn_tr_fpr, t_knn_tr_tpr)
        tr_t_auc.append(float(t_knn_tr_roc_auc))

        r_knn_tr_prob = r_knn.predict(tr_feature)
        r_knn_tr_pred = np.array(r_knn_tr_prob)
        r_knn_tr_fpr, r_knn_tr_tpr, r_knn_tr_thresholds = roc_curve(r_tmp_tr_y, r_knn_tr_pred)
        r_knn_tr_roc_auc = "%.3f" % auc(r_knn_tr_fpr, r_knn_tr_tpr)
        tr_r_auc.append(float(r_knn_tr_roc_auc))

        ent_knn_tr_prob = ent_knn.predict(tr_feature)
        ent_knn_tr_pred = np.array(ent_knn_tr_prob)
        ent_knn_tr_fpr, ent_knn_tr_tpr, ent_knn_tr_thresholds = roc_curve(ent_tmp_tr_y, ent_knn_tr_pred)
        ent_knn_tr_roc_auc = "%.3f" % auc(ent_knn_tr_fpr, ent_knn_tr_tpr)
        tr_ent_auc.append(float(ent_knn_tr_roc_auc))

        g_knn_tr_prob = g_knn.predict(tr_feature)
        g_knn_tr_pred = np.array(g_knn_tr_prob)
        g_knn_tr_fpr, g_knn_tr_tpr, g_knn_tr_thresholds = roc_curve(g_tmp_tr_y, g_knn_tr_pred)
        g_knn_tr_roc_auc = "%.3f" % auc(g_knn_tr_fpr, g_knn_tr_tpr)
        tr_g_auc.append(float(g_knn_tr_roc_auc))

        e_knn_tr_prob = e_knn.predict(tr_feature)
        e_knn_tr_pred = np.array(e_knn_tr_prob)
        e_knn_tr_fpr, e_knn_tr_tpr, e_knn_tr_thresholds = roc_curve(e_tmp_tr_y, e_knn_tr_pred)
        e_knn_tr_roc_auc = "%.3f" % auc(e_knn_tr_fpr, e_knn_tr_tpr)
        tr_e_auc.append(float(e_knn_tr_roc_auc))


        #test auc
        s_knn_te_prob = s_knn.predict(te_feature)
        s_knn_te_pred = np.array(s_knn_te_prob)
        s_knn_fpr, s_knn_tpr, s_knn_thresholds = roc_curve(s_tmp_te_y, s_knn_te_pred)
        s_knn_roc_auc = "%.3f" % auc(s_knn_fpr, s_knn_tpr)
        te_s_auc.append(float(s_knn_roc_auc))

        t_knn_te_prob = t_knn.predict(te_feature)
        t_knn_te_pred = np.array(t_knn_te_prob)
        t_knn_fpr, t_knn_tpr, t_knn_thresholds = roc_curve(t_tmp_te_y, t_knn_te_pred)
        t_knn_roc_auc = "%.3f" % auc(t_knn_fpr, t_knn_tpr)
        te_t_auc.append(float(t_knn_roc_auc))

        r_knn_te_prob = r_knn.predict(te_feature)
        r_knn_te_pred = np.array(r_knn_te_prob)
        r_knn_te_fpr, r_knn_te_tpr, _ = roc_curve(r_tmp_te_y, r_knn_te_pred)
        r_knn_te_roc_auc = "%.3f" % auc(r_knn_te_fpr, r_knn_te_tpr)
        te_r_auc.append(float(r_knn_te_roc_auc))

        ent_knn_te_prob = ent_knn.predict(te_feature)
        ent_knn_te_pred = np.array(ent_knn_te_prob)
        ent_knn_te_fpr, ent_knn_te_tpr, _ = roc_curve(ent_tmp_te_y, ent_knn_te_pred)
        ent_knn_te_roc_auc = "%.3f" % auc(ent_knn_te_fpr, ent_knn_te_tpr)
        te_ent_auc.append(float(ent_knn_te_roc_auc))

        g_knn_te_prob = g_knn.predict(te_feature)
        g_knn_te_pred = np.array(g_knn_te_prob)
        g_knn_te_fpr, g_knn_te_tpr, g_knn_te_thresholds = roc_curve(g_tmp_te_y, g_knn_te_pred)
        g_knn_te_roc_auc = "%.3f" % auc(g_knn_te_fpr, g_knn_te_tpr)
        te_g_auc.append(float(g_knn_te_roc_auc))

        e_knn_te_prob = e_knn.predict(te_feature)
        e_knn_te_pred = np.array(e_knn_te_prob)
        e_knn_te_fpr, e_knn_te_tpr, e_knn_te_thresholds = roc_curve(e_tmp_te_y, e_knn_te_pred)
        e_knn_te_roc_auc = "%.3f" % auc(e_knn_te_fpr, e_knn_te_tpr)
        te_e_auc.append(float(e_knn_te_roc_auc))

        if test == 2:
            #plt.title("ROC curve of Christmas")
            plt.figure(figsize=(9,8))
            x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            x_label = ["0.0", '0.2', "0.4", "0.6", '0.8', "1.0"]
            y = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            y_label = ["0.0", '0.2', "0.4", "0.6", '0.8', "1.0"]
            plt.plot(s_knn_fpr, s_knn_tpr, 'C4', label = 'Shopping', linewidth=2)
            plt.plot(t_knn_fpr, t_knn_tpr, 'C5', label = 'Travel', linewidth=2)
            plt.plot(r_knn_te_fpr, r_knn_te_tpr, 'C6', label = 'Restaurant and Dining', linewidth=2)
            plt.plot(ent_knn_te_fpr, ent_knn_te_tpr, 'C1', label = 'Entertainment', linewidth=2)
            plt.plot(g_knn_te_fpr, g_knn_te_tpr, 'C2', label = 'Games', linewidth=2)
            plt.plot(e_knn_te_fpr, e_knn_te_tpr, 'C3', label = 'Education', linewidth=2)
            plt.legend(loc = 'lower right', fontsize=15)
            plt.plot([0, 1], [0, 1],'r--', linewidth=0.5)
            plt.xticks(x, x_label, fontsize=25)
            plt.yticks(y, y_label, fontsize=25)
            plt.ylabel('True Positive Rate', fontsize=25)
            plt.xlabel('False Positive Rate', fontsize=25)
            plt.savefig("SVM.pdf", format='pdf', dpi=1000)
            plt.show()
        '''

    print("svm_training_f1")
    print("Shopping:  " + str("%.3f" % np.mean(tr_s_f1)))
    print("Travel:  " + str("%.3f" % np.mean(tr_t_f1)))
    print("Restaurant and Dining:  " + str("%.3f" % np.mean(tr_r_f1)))
    print("Entertainment:  " + str("%.3f" % np.mean(tr_ent_f1)))
    print("Games:  " + str("%.3f" % np.mean(tr_g_f1)))
    print("Education:  " + str("%.3f" % np.mean(tr_e_f1)))
    print("")
    print("svm_test_f1")
    print("Shopping:  " + str("%.3f" % np.mean(te_s_f1)))
    print("Travel:  " + str("%.3f" % np.mean(te_t_f1)))
    print("Restaurant and Dining:  " + str("%.3f" % np.mean(te_r_f1)))
    print("Entertainment:  " + str("%.3f" % np.mean(te_ent_f1)))
    print("Games:  " + str("%.3f" % np.mean(te_g_f1)))
    print("Education:  " + str("%.3f" % np.mean(te_e_f1)))

    '''
    print("svm_training_auc")
    print("Shopping:  " + str("%.3f" % np.mean(tr_s_auc)))
    print("Travel:  " + str("%.3f" % np.mean(tr_t_auc)))
    print("Restaurant and Dining:  " + str("%.3f" % np.mean(tr_r_auc)))
    print("Entertainment:  " + str("%.3f" % np.mean(tr_ent_auc)))
    print("Games:  " + str("%.3f" % np.mean(tr_g_auc)))
    print("Education:  " + str("%.3f" % np.mean(tr_e_auc)))
    print("")
    print("svm_test_auc")
    print("Shopping:  " + str("%.3f" % np.mean(te_s_auc)))
    print("Travel:  " + str("%.3f" % np.mean(te_t_auc)))
    print("Restaurant and Dining:  " + str("%.3f" % np.mean(te_r_auc)))
    print("Entertainment:  " + str("%.3f" % np.mean(te_ent_auc)))
    print("Games:  " + str("%.3f" % np.mean(te_g_auc)))
    print("Education:  " + str("%.3f" % np.mean(te_e_auc)))
    '''







if __name__ == '__main__':
    main()

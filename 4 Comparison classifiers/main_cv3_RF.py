from CV import kfold

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve,auc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import gpboost as gpb


time = 20
k = 5
for mm in range(2,3):
    for cv in range(3, 4):
        data_file = './data' + str(mm) + '/data21.csv'
        label_file = './data' + str(mm) + '/label.csv'
        data = pd.read_csv(data_file, header=None, index_col=None).to_numpy()
        label = pd.read_csv(label_file, index_col=None, header=None).to_numpy()
        label_copy = label.copy()
        row, col = label.shape
        if cv == 4:
            c = np.array([(i, j) for i in range(row) for j in range(col)])
        else:
            a = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j]])
            b = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j] == 0])
            np.random.shuffle(b)
            sample = len(a)
            b = b[:sample]
        mPREs = np.array([])
        mACCs = np.array([])
        mRECs = np.array([])
        mAUCs = np.array([])
        mAUPRs = np.array([])
        mF1 = np.array([])

        for j in range(time):
            if cv == 4:
                c_tr, c_te = np.array(kfold(c, k=k, row=row, col=col, cv=cv))
            elif cv == 3:
                a_tr, a_te = np.array(kfold(a, k=k, row=row, col=col, cv=cv))
                b_tr, b_te = np.array(kfold(b, k=k, row=row, col=col, cv=cv))
            else:
                c = np.vstack([a, b])
                c_tr, c_te = np.array(kfold(c, k=k, row=row, col=col, cv=cv))
            for i in range(k):
                if cv == 4:
                    b_tr = []
                    a_tr = []
                    # print(c_tr[i])
                    for ep in c_tr[i]:
                        if label_copy[c[ep][0]][c[ep][1]]:
                            b_tr.append(c[ep])
                        else:
                            a_tr.append(c[ep])
                    b_te = []
                    a_te = []
                    for ep in c_te[i]:
                        if label_copy[c[ep][0]][c[ep][1]]:
                            b_te.append(c[ep])
                        else:
                            a_te.append(c[ep])
                    b_te = np.array(b_te)
                    b_tr = np.array(b_tr)
                    a_te = np.array(a_te)
                    a_tr = np.array(a_tr)
                    np.random.shuffle(b_te)
                    np.random.shuffle(a_te)
                    np.random.shuffle(b_tr)
                    np.random.shuffle(a_tr)
                    a_tr = a_tr[:len(b_tr)]
                    a_te = a_te[:len(b_te)]
                    train_sample = np.vstack([a_tr, b_tr])
                    test_sample = np.vstack([a_te, b_te])
                elif cv == 3:
                    train_sample = np.vstack([np.array(a[a_tr[i]]), np.array(b[b_tr[i]])])
                    test_sample = np.vstack([np.array(a[a_te[i]]), np.array(b[b_te[i]])])
                else:
                    train_sample = np.array(c[c_tr[i]])
                    test_sample = np.array(c[c_te[i]])
                train_land = train_sample[:, 0] * col + train_sample[:, 1]
                test_land = test_sample[:, 0] * col + test_sample[:, 1]
                np.random.shuffle(train_land)
                np.random.shuffle(test_land)

                X_tr = data[train_land][:, :-1]
                y_tr = data[train_land][:, -1]
                X_te = data[test_land][:, :-1]
                y_te = data[test_land][:, -1]

                model1 = RandomForestClassifier(n_estimators=200, max_depth=9)
                model1.fit(X_tr, y_tr)

                score = model1.predict_proba(X_te)[:,1]

                pred = []
                prob =  score
                for n in prob:
                    if n > 0.5:
                        pred.append(1)
                    else:
                        pred.append(0)
                pre_label = np.array(pred)

                fpr, tpr, threshold = roc_curve(y_te, score)
                pre, rec_, _ = precision_recall_curve(y_te, score)

                acc = accuracy_score(y_te, pre_label)
                rec = recall_score(y_te, pre_label)
                f1 = f1_score(y_te, pre_label)
                Pre = precision_score(y_te, pre_label)
                au = auc(fpr, tpr)
                apr = auc(rec_, pre)
                mPREs = np.append(mPREs, Pre)
                mACCs = np.append(mACCs, acc)
                mRECs = np.append(mRECs, rec)
                mAUCs = np.append(mAUCs, au)
                mAUPRs = np.append(mAUPRs, apr)
                mF1 = np.append(mF1, f1)

                curve_1 = np.vstack([fpr, tpr])
                curve_1 = pd.DataFrame(curve_1.T)
                curve_1.to_csv('./result_d2_cv3_rf/d' + str(mm) + 'c' + str(cv) + '_c' + str(au) + '.csv',
                               header=None, index=None)

                curve_2 = np.vstack([rec_, pre])
                curve_2 = pd.DataFrame(curve_2.T)
                curve_2.to_csv('./result_d2_cv3_rf/d' + str(mm) + 'c' + str(cv) + '_r' + str(apr) + '.csv',
                               header=None, index=None)

                print('Precision is :{}'.format(Pre))
                print('Recall is :{}'.format(rec))
                print("ACC is: {}".format(acc))
                print("F1 is: {}".format(f1))
                print("AUC is: {}".format(au))
                print('AUPR is :{}'.format(apr))

            print("One time 5-Folds computed.")

        toa = np.vstack([mAUCs, mAUPRs])
        toa = pd.DataFrame(toa)
        toa.to_csv('./result_d2_cv3_rf/' + 'cv' + str(cv) + '_data' + str(mm) + '.csv', header=None, index=None)
        PRE = mPREs.mean()
        ACC = mACCs.mean()
        REC = mRECs.mean()
        AUC = mAUCs.mean()
        AUPR = mAUPRs.mean()
        F1 = mF1.mean()

        PRE_err = np.std(mPREs)
        ACC_err = np.std(mACCs)
        REC_err = np.std(mRECs)
        AUC_err = np.std(mAUCs)
        AUPR_err = np.std(mAUPRs)
        F1_err = np.std(mF1)

        print('\n')
        print("PRE is:{}±{}".format(round(PRE, 4), round(PRE_err, 4)))
        print("REC is:{}±{}".format(round(REC, 4), round(REC_err, 4)))
        print("ACC is:{}±{}".format(round(ACC, 4), round(ACC_err, 4)))
        print("F1 is:{}±{}".format(round(F1, 4), round(F1_err, 4)))
        print('AUC is :{}±{}'.format(round(AUC, 4), round(AUC_err, 4)))
        print('AUPR is :{}±{}'.format(round(AUPR, 4), round(AUPR_err, 4)))

        f_1 = open('./result_d2_cv3_rf/' + 'res_cv.txt', 'a')
        f_1.write('data' + str(mm) + 'cv' + str(cv) + ':\n')
        f_1.write(str(round(PRE, 4)) + '±' + str(round(PRE_err, 4)) + '\t')
        f_1.write(str(round(REC, 4)) + '±' + str(round(REC_err, 4)) + '\t')
        f_1.write(str(round(ACC, 4)) + '±' + str(round(ACC_err, 4)) + '\t')
        f_1.write(str(round(F1, 4)) + '±' + str(round(F1_err, 4)) + '\t')
        f_1.write(str(round(AUC, 4)) + '±' + str(round(AUC_err, 4)) + '\t')
        f_1.write(str(round(AUPR, 4)) + '±' + str(round(AUPR_err, 4)) + '\t\n\n')
        f_1.close()
        print('\nwrite\n')


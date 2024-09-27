# ======================================================================
#
# -*- coding: utf-8 -*-
#
# ======================================================================

# 'data' is the sample to be tested, 'row' is the number of rows, 'col' is the number of columns
# CV1 represents rows, CV2 represents columns

import numpy as np
from sklearn.model_selection import KFold


def kfold(data, k, row=0, col=0, cv=4):
    dlen = len(data)
    if cv != 4:
        if cv == 1:
            lens = row
        elif cv == 2:
            lens = col
        else:
            lens = dlen
        test_res = []
        train_res = []
        d = list(range(lens))
        kf = KFold(k, shuffle=True)
        d = kf.split(d)
        for i in d:
            test_res.append(list(i[1]))
            train_res.append(list(i[0]))
        if cv == 3:
            return train_res, test_res
        else:
            train_s = []
            test_s = []
            for i in range(k):
                train_ = []
                test_ = []
                for j in range(dlen):
                    if data[j][cv - 1] in test_res[i]:
                        test_.append(j)
                    else:
                        train_.append(j)
                train_s.append(train_)
                test_s.append(test_)
            return train_s, test_s
    else:
        r = list(range(row))
        c = list(range(col))
        kf = KFold(k, shuffle=True)
        r = kf.split(r)
        c = kf.split(c)
        r_test_res = []
        r_train_res = []
        c_test_res = []
        c_train_res = []
        for i in r:
            r_test_res.append(list(i[1]))
            r_train_res.append(list(i[0]))
        for i in c:
            c_test_res.append(list(i[1]))
            c_train_res.append(list(i[0]))
        train_s = []
        test_s = []
        for i in range(k):
            train_ = []
            test_ = []
            for m in range(dlen):
                flag_1 = False
                flag_2 = False
                if data[m][0] in r_test_res[i]:
                    flag_1 = True
                if data[m][1] in c_test_res[i]:
                    flag_2 = True
                if flag_1 and flag_2:
                    test_.append(m)
                if (not flag_1) and (not flag_2):
                    train_.append(m)
            train_s.append(train_)
            test_s.append(test_)
        return train_s, test_s


def get_one_hot(targets, nb_classes) -> object:
    """
    :rtype: object
    """
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])

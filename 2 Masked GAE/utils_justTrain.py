import torch
import random
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_metrics(y_true, y_pred):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP*TN-FP*FN)/np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    F1_score = 2*(precision*sensitivity)/(precision+sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


def get_data(data_ID, output_dim):
    lncRNA = np.loadtxt('data/dataset1/lnc_Similarity_55.csv',delimiter=',')  # (541, 541)
    Dis = np.loadtxt('data/dataset1/dis_Similarity_55.csv',delimiter=',')  # (831, 831)
    association = np.loadtxt('data/dataset1/label.csv',delimiter=',')  # (541, 831)   812*780
    if data_ID == 2:
        lncRNA = np.loadtxt('data/dataset2/lnc_Similarity_55.csv', delimiter=',')  # (286, 286)
        Dis = np.loadtxt('data/dataset2/dis_Similarity_55.csv', delimiter=',')  # (39, 39)
        association = np.loadtxt('data/dataset2/label.csv', delimiter=',')  # (286, 39)

    l_emb = []
    for m in range(len(lncRNA)):
        l_emb.append(lncRNA[m].tolist())
    l_emb = [lst + [0] * (output_dim - len(l_emb[0])) for lst in l_emb]
    l_emb = torch.Tensor(l_emb)

    d_emb = []
    for s in range(len(Dis)):
        d_emb.append(Dis[s].tolist())
    d_emb = [lst + [0] * (output_dim - len(d_emb[0])) for lst in d_emb]
    d_emb = torch.Tensor(d_emb)

    feature = torch.cat([l_emb, d_emb])

    adj = []
    for m in range(len(lncRNA)):
        for s in range(len(Dis)):
            if association[m][s] == 1:
                adj.append([m, s + len(lncRNA)])
    adj = torch.LongTensor(adj).T
    data = Data(x=feature, edge_index=adj).cuda()

    train_data, _, test_data = T.RandomLinkSplit(num_val=0, num_test=0,
                                                 is_undirected=True, split_labels=False,
                                                 add_negative_train_samples=False)(data)

    splits = dict(train=train_data, test=test_data)
    return splits


if __name__ == '__main__':
    data = get_data(2, 1024)

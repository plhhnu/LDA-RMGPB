import numpy as np

l_l = np.loadtxt('./dataset2/SVD/lda_u_2.csv',delimiter=',')
n_l = np.loadtxt('./dataset2/MGAE/2lnc_GAE_features8.txt',delimiter=',')
l_d = np.loadtxt('./dataset2/SVD/lda_v_2.csv',delimiter=',')
n_d = np.loadtxt('./dataset2/MGAE/2dis_GAE_features8.txt',delimiter=',')

lncrna_feature = np.hstack((l_l,n_l))
disease_feature = np.hstack((l_d,n_d))



np.save('./dataset2/merge/rna10_feature.npy', lncrna_feature)
np.save('./dataset2/merge/dis10_feature.npy', disease_feature)

print("拼接完成！")




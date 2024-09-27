import argparse
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--q', default=5, type=int, help='rank')
args = parser.parse_args()
matrix_data = pd.read_csv('label.csv', header=None).values

sparse_matrix = sp.csr_matrix(matrix_data)

torch_sparse_matrix = torch.tensor(sparse_matrix.todense(), dtype=torch.float32).to_sparse().coalesce()

print('Performing SVD...')
svd_u, s, svd_v = torch.svd_lowrank(torch_sparse_matrix, q=args.q)
u_mul_s = svd_u @ torch.diag(s)
v_mul_s = svd_v @ torch.diag(s)
print('SVD done.')
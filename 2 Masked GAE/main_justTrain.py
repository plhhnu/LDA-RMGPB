import torch
import argparse
from utils_justTrain import get_data, set_seed
from model_justTrain import GNNEncoder, EdgeDecoder, DegreeDecoder, MGAE, Mask
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# main parameter
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=int, default=2, help="Choose Datasets (1 or 2)")
parser.add_argument('--seed', type=int, default=300, help="Random seed for model and dataset.")
parser.add_argument('--dim', type=int, default=300, help='Feature Dimension of Similarity Matrix ')
parser.add_argument('--alpha', type=float, default=0.007, help='loss weight for degree prediction.')
parser.add_argument('--p', type=float, default=0.4, help='Mask ratio')
args = parser.parse_args()
set_seed(args.seed)

splits = get_data(args.dataset, args.dim)
encoder = GNNEncoder(in_channels=args.dim, hidden_channels=64, out_channels=8)
edge_decoder = EdgeDecoder(in_channels=8, hidden_channels=64, out_channels=1)
degree_decoder = DegreeDecoder(in_channels=8, hidden_channels=64, out_channels=1)
mask = Mask(p=args.p)

model = MGAE(encoder, edge_decoder, degree_decoder, mask).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)#lr=0.001, weight_decay=5e-5
for epoch in range(1000):
    model.train()
    loss = model.train_epoch(splits['train'], optimizer, alpha=args.alpha)
    print('Epoch:', epoch, 'Loss:', loss)
model.eval()
z = model.encoder(splits['train'].x, splits['train'].edge_index).detach().cpu()
zlnc = z[:89, :]
zdis = z[89:, :]
np.savetxt('2lnc_GAE_features8.txt', zlnc, delimiter=',',)
np.savetxt('2dis_GAE_features8.txt', zdis, delimiter=',',)

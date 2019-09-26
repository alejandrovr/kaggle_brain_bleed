from code.net import BleedNet

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pickle
import random
import glob
from copy import deepcopy
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from code.net import BleedNet
from code.featurize import next_batch, PF_Loader
import torch.nn as nn
import torch.optim as optim
import psutil
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
#df = pd.read_csv('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/sample.csv')
df['Sub_type'] = df['ID'].str.split("_", n = 3, expand = True)[2]
df['PatientID'] = df['ID'].str.split("_", n = 3, expand = True)[1]
bleed_subtype_df = df.loc[df['Sub_type'] == 'any']


df_subtype_pos = bleed_subtype_df.loc[bleed_subtype_df['Label'] == 1]
df_subtype_neg = bleed_subtype_df.loc[bleed_subtype_df['Label'] == 0]

pf_loader_pos = PF_Loader(df_subtype_pos)
pf_loader_neg = PF_Loader(df_subtype_neg)

n_batches = 3000
batch_size = 20
lr = 0.1
lr_log = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = nn.CrossEntropyLoss()
bleed_net = BleedNet()
bleed_net.to(device)

optimizer = optim.SGD(bleed_net.parameters(), lr=lr) #momentum?
from torch.optim.lr_scheduler import StepLR
stepsize = 200
lr_gamma = 0.99
scheduler = StepLR(optimizer, step_size=stepsize, gamma=lr_gamma)
loss_log = []
val_loss_log = []
test_loss_log = []


#TRAIN THE MODEL
for i in range(n_batches):
    bleed_net.train()
    try:
        x, y = next_batch(pf_loader_pos,pf_loader_neg,batch_size=batch_size)
    except:
        continue
    x_train_tensor = torch.from_numpy(x).float().to(device)
    y_train_tensor = torch.from_numpy(y).long().to(device)
    y_train_tensor = y_train_tensor.argmax(dim=1)
    yhat = bleed_net(x_train_tensor)
    yhat_choice = yhat.argmax(dim=1)

    acc = y_train_tensor == yhat_choice
    acc = acc.sum().float() / acc.shape[0]
    loss = loss_fn(yhat, y_train_tensor)
    loss.backward()    
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    print('Loss: {} | Acc: {} | Batch {}/{}'.format(loss.item(),acc,i,n_batches))


import time
timestr = time.strftime("%Y%m%d-%H%M%S")
torch.save(bleed_net.state_dict(), 'models/bleednet_acc_{}_{}.torch'.format(acc,timestr))














from code.net import BleedNet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from code.net import BleedNet, BleedNet2
from code.featurize import next_batch, PF_Loader
import torch.nn as nn
import torch.optim as optim
import psutil
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

def df2pf_loader(df):
    df['Sub_type'] = df['ID'].str.split("_", n = 3, expand = True)[2]
    df['PatientID'] = df['ID'].str.split("_", n = 3, expand = True)[1]
    bleed_subtype_df = df.loc[df['Sub_type'] == 'any']

    df_subtype_pos = bleed_subtype_df.loc[bleed_subtype_df['Label'] == 1]
    df_subtype_neg = bleed_subtype_df.loc[bleed_subtype_df['Label'] == 0]

    pf_loader_pos = PF_Loader(df_subtype_pos)
    pf_loader_neg = PF_Loader(df_subtype_neg)
    return pf_loader_pos, pf_loader_neg


make_split = False

if make_split:
    df = pd.read_csv('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
    df = shuffle(df)

    msk = np.random.rand(len(df)) < 0.8 #80% for training
    train = df[msk]
    val_test = df[~msk]

    msk_val_test = np.random.rand(len(val_test)) < 0.5
    val = val_test[msk_val_test] #10% val
    test = val_test[~msk_val_test] #10% test

    print('Train size:', len(train))
    print('Val size:', len(val))
    print('Test size:', len(test))

    train.to_csv('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/split_train.csv')
    val.to_csv('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/split_val.csv')
    test.to_csv('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/split_test.csv')
    df = train

else:
    df = pd.read_csv('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/split_train.csv')
    val = pd.read_csv('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/split_val.csv')
    test = pd.read_csv('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/split_test.csv')

#Load data
train_pf_loader_pos, train_pf_loader_neg = df2pf_loader(df.sample(1000))


val_pf_loader_pos, val_pf_loader_neg = df2pf_loader(val) 
test_pf_loader_pos, test_pf_loader_neg = df2pf_loader(test) 

#Learning and net parameters
n_batches = 500
batch_size = 10
lr = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = nn.CrossEntropyLoss()
bleed_net = BleedNet2()
bleed_net.to(device)
optimizer = optim.SGD(bleed_net.parameters(), lr=lr) #momentum?
from torch.optim.lr_scheduler import StepLR
stepsize = 100 #100
lr_gamma = 0.9
scheduler = StepLR(optimizer, step_size=stepsize, gamma=lr_gamma)

#Initialize logs
train_loss_log = []
val_loss_log = []
test_loss_log = []


#TRAIN THE MODEL
for i in range(n_batches):
    bleed_net.train()
    x, y = next_batch(train_pf_loader_pos,train_pf_loader_neg,batch_size=batch_size)
    x_train_tensor = torch.from_numpy(x).float().to(device)
    y_train_tensor = torch.from_numpy(y).long().to(device)
    y_train_tensor = y_train_tensor.argmax(dim=1)
    yhat = bleed_net(x_train_tensor)
    yhat_choice = yhat.argmax(dim=1)

    acc = y_train_tensor == yhat_choice
    acc = acc.sum().float() / acc.shape[0]
    acc = acc.cpu().item()
    loss = loss_fn(yhat, y_train_tensor)
    loss.backward()    
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    print('Loss: {} | Acc: {} | Batch {}/{}'.format(loss.item(),acc,i,n_batches))
    train_loss_log.append((loss.item(),acc))

    if i % 100 == 0:
        bleed_net.eval()
        try:
            x, y = next_batch(val_pf_loader_pos, val_pf_loader_neg, batch_size=25)
        except:
            continue
        x_val_tensor = torch.from_numpy(x).float().to(device)
        y_val_tensor = torch.from_numpy(y).long().to(device)
        y_val_tensor = y_val_tensor.argmax(dim=1)
        yhat = bleed_net(x_val_tensor)
        yhat_choice = yhat.argmax(dim=1)

        acc = y_val_tensor == yhat_choice
        acc = acc.sum().float() / acc.shape[0]
        acc = acc.cpu().item()
        loss = loss_fn(yhat, y_val_tensor)  
        optimizer.zero_grad()
        print('\n\n\nVALIDATION Loss: {} | Acc: {} | Batch {}/{}\n\n\n'.format(loss.item(),acc,i,n_batches))
        val_loss_log.append((loss.item(),acc))


#FINALLY, TEST IT
print('Evaluating net performance on test split...')
bleed_net.eval()
for test_idx in range(100):
    x, y = next_batch(test_pf_loader_pos, test_pf_loader_neg, batch_size=10)
    x_test_tensor = torch.from_numpy(x).float().to(device)
    y_test_tensor = torch.from_numpy(y).long().to(device)
    y_test_tensor = y_test_tensor.argmax(dim=1)
    yhat = bleed_net(x_test_tensor)
    yhat_choice = yhat.argmax(dim=1)
    acc = y_test_tensor == yhat_choice
    acc = acc.sum().float() / acc.shape[0]
    acc = acc.cpu().item()
    loss = loss_fn(yhat, y_test_tensor)  
    optimizer.zero_grad()
    #print('\n\n\nTEST Loss: {} | Acc: {} | Batch {}/{}\n\n\n'.format(loss.item(),acc,i,n_batches))
    test_loss_log.append((loss.item(),acc))

print([i[1] for i in test_loss_log])
print('Saving the model...')
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
torch.save(bleed_net.state_dict(), 'models/bleednet_testacc_{}_{}.torch'.format(acc,timestr))
acc = int(np.mean([i[1] for i in test_loss_log]) * 100)
print('Model saved in:','models/bleednet_testacc_{}_{}.torch'.format(acc,timestr))

plt.plot([i[0] for i in train_loss_log],color='blue')
plt.plot([i[1] for i in train_loss_log],color='red')
#plt.plot([i[1] for i in val_loss_log])
#plt.plot([i[1] for i in test_loss_log])
plt.show()
print('Exiting...')











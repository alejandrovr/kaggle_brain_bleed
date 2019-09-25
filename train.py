from code.net import BleedNet

import os
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
from rDeep.net import KDeepLight, KDeepLighter
from rDeep.featurize import featurize_prot_lig, LJ_potential
import torch.nn as nn
import torch.optim as optim
from multiprocessing import Pool
import psutil
import pickle
import os
from rDeep.kdeeploader import PF_Loader
import matplotlib.pyplot as plt

def next_batch_category(codes, batch_size=300):
    pocket_batch = []
    for i in range(batch_size):
        new_code = random.choice(codes)
        try:
            pose, label = pf_loader.__getitem__(new_code)
            pocket_batch.append([pose,label]) 
        except Exception as e:
            print(new_code,e)
            continue
    random.shuffle(pocket_batch)   
    
    #find amound of examples in less pop class
    pb_count = {tuple(label[0]):0 for _,label in pocket_batch}
    for pb in pocket_batch:
        pb_count[tuple(pb[1][0])] += 1
        
    less_pop = min([i[1] for i in pb_count.items()])
    
    class_eq_count = {tuple(label[0]):0 for _,label in pocket_batch}
    final_batch = []
    for texample in pocket_batch:
        class_key = tuple(texample[1][0])
        if class_eq_count[class_key] < less_pop:
            final_batch.append(texample)
            class_eq_count[class_key] += 1
            
    return final_batch


pf_loader = PF_Loader(rotation_mode="random", rotate=True, perturb=False,pretrain=pretrain_model)


n_batches = 3000
batch_size = 20
lr = 0.1
lr_log = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()
kdeep_net = KDeepLighter()

if not pretrain_model:
    kdeep_net.load_state_dict(torch.load('models/rDeep_val_91_20190925-105400_True.torch'))
    kdeep_net.wrap_up[2] = nn.Linear(in_features=512, out_features=1, bias=True)

#Best so far: trained in 30000 epochs, crystal vs poses models/rDeep_val_84_20190924-174225.pkl
kdeep_net.to(device)

optimizer = optim.SGD(kdeep_net.parameters(), lr=lr) #momentum?
from torch.optim.lr_scheduler import StepLR
stepsize = 200
lr_gamma = 0.90
scheduler = StepLR(optimizer, step_size=stepsize, gamma=lr_gamma)
loss_log = []
val_loss_log = []
test_loss_log = []



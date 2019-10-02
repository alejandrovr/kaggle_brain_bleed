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
from code.featurize import next_batch, PF_Loader, dcm2np
import torch.nn as nn
import torch.optim as optim
import psutil
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle


device = 'cuda' if torch.cuda.is_available() else 'cpu'

['intraparenchymal', 'any', 'epidural', 'intraventricular','subarachnoid', 'subdural']

#any
bleed_net_any = BleedNet2()
bleed_net_any.load_state_dict(torch.load('models/bleednet_any_testacc_79_20191001-122813.torch'))
bleed_net_any.to(device)
bleed_net_any.eval()

#intra
bleed_net_intra = BleedNet2()
bleed_net_intra.load_state_dict(torch.load('models/bleednet_intraparenchymal_testacc_78_20191001-115643.torch'))
bleed_net_intra.to(device)
bleed_net_intra.eval()

#subarach
bleed_net_subar = BleedNet2()
bleed_net_subar.load_state_dict(torch.load('models/bleednet_subarachnoid_testacc_73_20191001-124700.torch'))
bleed_net_subar.to(device)
bleed_net_subar.eval()

#subdural
bleed_net_subdural = BleedNet2()
bleed_net_subdural.load_state_dict(torch.load('models/bleednet_subdural_testacc_75_20191001-142209.torch'))
bleed_net_subdural.to(device)
bleed_net_subdural.eval()

#intraven
bleed_net_intraven = BleedNet2()
bleed_net_intraven.load_state_dict(torch.load('models/bleednet_intraventricular_testacc_82_20191001-124024.torch'))
bleed_net_intraven.to(device)
bleed_net_intraven.eval()

#epidural
bleed_net_epi = BleedNet2()
bleed_net_epi.load_state_dict(torch.load('models/bleednet_epidural_testacc_80_20191001-123408.torch'))
bleed_net_epi.to(device)
bleed_net_epi.eval()

#sample submission
#ID_28fbab7eb_epidural,0.5
#ID_28fbab7eb_intraparenchymal,0.5
#ID_28fbab7eb_intraventricular,0.5
#ID_28fbab7eb_subarachnoid,0.5
#ID_28fbab7eb_subdural,0.5
#ID_28fbab7eb_any,0.5

nets = [bleed_net_epi,bleed_net_intra,bleed_net_intraven,bleed_net_subar,bleed_net_subdural,bleed_net_any]

images_path = '/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'  
test_csv = '/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv' 
df = pd.read_csv(test_csv)
image_scores = []
totalrows = len(df)
ids_done = []
for idxrow, row in df.iterrows():
    print(idxrow,'/',totalrows)
    fullid = row.ID[:row.ID.rfind('_')] #ID_28fbab7eb_epidural
    if fullid in ids_done:
        continue

    image = images_path + fullid + '.dcm'
    npimage = dcm2np(image)
    if npimage.shape == (512,512):
        npimage = npimage[np.newaxis,np.newaxis,:]
        x_train_tensor = torch.from_numpy(npimage).float().to(device)
        for net in nets:
            yhat = net(x_train_tensor)
            yhat_choice = yhat.argmax(dim=1).item()
            image_scores.append((fullid, float(yhat_choice)))
    else:
        for i in nets:
            image_scores.append((fullid, 0.5))

    ids_done.append(fullid)
      


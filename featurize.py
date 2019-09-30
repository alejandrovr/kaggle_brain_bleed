import pydicom
import dicom_numpy
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from torch.utils.data import Dataset

def dcm2np(dcm_file):
    ds = pydicom.dcmread(dcm_file)
    im = ds.pixel_array
    try:
        im = im / im.max()
    except:
        print('Something went wrong!')
        im = im * 0 
    return im

def next_batch(pf_loader_p, pf_loader_n, batch_size=100):
    batch = []
    for i in range(batch_size):
        if i % 2 == 0:
            x, y = pf_loader_p.__getitem__(pos_neg=1)
        else:
            x, y = pf_loader_n.__getitem__(pos_neg=0)
        #input('next?')
        if x.shape == (512, 512):
            batch.append((x,y))

    batch_x = np.array([i[0] for i in batch])
    batch_x = batch_x[:,np.newaxis,:]
    batch_y = np.array([[0,1] if i[1]==1 else [1,0] for i in batch])

    return batch_x, batch_y

class PF_Loader(Dataset):
    def __init__(self, df):
        """Constructor for Loader"""
        self.df = df

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, pos_neg=0):
        """Itemgetter for Loader"""
        data = self.df.sample(1)
        img_name = data.iloc[0].PatientID
        file_name = '/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_'+img_name+'.dcm'
        np_image = dcm2np(file_name)
        label = pos_neg
        #plt.imshow(np_image, cmap=plt.cm.bone)
        #plt.title(str(label))
        #plt.show()
        return np_image, label


if __name__ == "__main__":

    #thanks to https://www.kaggle.com/marcovasquez/basic-eda-data-visualization
    df = pd.read_csv('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
    df['Sub_type'] = df['ID'].str.split("_", n = 3, expand = True)[2]
    df['PatientID'] = df['ID'].str.split("_", n = 3, expand = True)[1]

    if False:
        for subtype in ['intraparenchymal', 'any', 'epidural', 'intraventricular','subarachnoid', 'subdural']:
            bleed_subtype_df = df.loc[df['Sub_type'] == subtype]
            df_subtype_pos = bleed_subtype_df.loc[bleed_subtype_df['Label'] == 1]
            df_subtype_neg = bleed_subtype_df.loc[bleed_subtype_df['Label'] == 0]

            rows_subtype_pos = df_subtype_pos.sample(5)
            rows_subtype_neg = df_subtype_neg.sample(5)

            for idx, row in rows_subtype_pos.iterrows():
                file_name = '/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_'+row.PatientID+'.dcm'
                print(file_name)
                #plt.imshow(dcm2np(file_name), cmap=plt.cm.bone)
                plt.imshow(dcm2np(file_name))
                plt.title('subtype_{}_{}'.format(subtype,'+'))
                plt.show()

            for idx, row in rows_subtype_neg.iterrows():
                file_name = '/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_'+row.PatientID+'.dcm'
                print(file_name)
                plt.imshow(dcm2np(file_name), cmap=plt.cm.bone)
                plt.title('subtype_{}_{}'.format(subtype,'-'))
                plt.show()


            input('Next subtype?')



import pydicom
import dicom_numpy
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os

def dcm2np(dcm_file):
    ds = pydicom.dcmread(dcm_file)
    im = ds.pixel_array
    return im

if __name__ == "__main__":
    #thanks to https://www.kaggle.com/marcovasquez/basic-eda-data-visualization
    dcim_files = glob.glob('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_2*')
    df = pd.read_csv('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
    df['Sub_type'] = df['ID'].str.split("_", n = 3, expand = True)[2]
    df['PatientID'] = df['ID'].str.split("_", n = 3, expand = True)[1]

    for subtype in ['any', 'epidural', 'intraparenchymal', 'intraventricular','subarachnoid', 'subdural']:
        df_subtype = df.loc[(df['Label'] == 1) & (df['Sub_type'] == subtype)]
        rows_subtype = df_subtype.sample(1)
        for idx, row in rows_subtype.iterrows():
            file_name = '/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_'+row.PatientID+'.dcm'
            print(file_name)
            plt.imshow(dcm2np(file_name), cmap=plt.cm.bone)
            plt.show()

        input('Next subtype?')


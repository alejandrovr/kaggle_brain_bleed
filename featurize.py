import pydicom
import dicom_numpy

def extract_voxel_data(list_of_dicom_files):
    datasets = [pydicom.read_file(f) for f in list_of_dicom_files]
    try:
        voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(datasets)
    except dicom_numpy.DicomImportException as e:
        # invalid DICOM data
        raise
    return voxel_ndarray


if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt
    dcim_files = glob.glob('/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_2a77db2aa*')
    #['/home/alejandro/kgl/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_2a77db2aa.dcm']
    ds = pydicom.dcmread(dcim_files[0])
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.show()




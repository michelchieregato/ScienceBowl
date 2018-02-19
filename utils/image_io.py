# coding: utf-8
import os
import sys
import shutil
import re
import random
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from time import time
from datetime import datetime, date
from itertools import chain

from dicom.dataset import Dataset, FileDataset

from nibabel import Nifti1Image, save

from glob import glob

from scipy.misc import imresize
from skimage.io import imread

import dicom
from dicom.tag import Tag
from dicom import filewriter, read_file

try:
    from sklearn.model_selection import KFold, ShuffleSplit
except ImportError:
    from sklearn.cross_validation import KFold, ShuffleSplit


def get_im_sitk(path):
    """Get simpleitk image from path."""
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    return img


def load_serie(path, slice_by_slice=False):
    metadata = {}

    reader = sitk.ImageSeriesReader()
    reader.SetOutputPixelType(sitk.sitkFloat32)
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)

    img = sitk.ReadImage(dicom_names[0])

    if not slice_by_slice:
        vol = reader.Execute()
    else:
        vol = [sitk.GetArrayFromImage(sitk.ReadImage(d))[0] for d in dicom_names]
        vol = np.asarray(vol)
        vol = sitk.GetImageFromArray(vol)

    dicom_metadata = read_file(dicom_names[0])
    # del dicom_metadata[Tag(['6000','3000'])] #Del the pixel array data from metadata

    metadata['dicom_metadata'] = dicom_metadata
    metadata['SOP_class_uid'] = img.GetMetaData('0008|0016')
    metadata['SOP_instance_uid'] = img.GetMetaData('0008|0018')
    metadata['modality'] = img.GetMetaData('0008|0060')
    metadata['study_instance_uid'] = img.GetMetaData('0020|000d')
    metadata['patient_birth_date'] = img.GetMetaData('0010|0030')
    try:
        metadata['patient_orientation'] = img.GetMetaData('0020|0037').split('\\')
    except:
        metadata['patient_orientation'] = None
        ADC_REGEX = '(ADC|adc)'
        parent_dir = os.path.abspath(os.path.join(path, os.pardir))
        for serie in os.listdir(parent_dir):
            if re.search(ADC_REGEX, serie):
                bval_path = glob(parent_dir + '/BVAL*/*')[0]
                bval = sitk.ReadImage(bval_path)
                metadata['patient_orientation'] = bval.GetMetaData('0020|0037').split('\\')

    metadata['pixel_spacing'] = [str(img.GetSpacing()[0]), str(img.GetSpacing()[1])]
    metadata['spacing_between_slices'] = str(img.GetSpacing()[2])

    metadata['serie_number'] = img.GetMetaData('0020|0011')

    try:
        metadata['slice_thickness'] = img.GetMetaData('0018|0050')
    except:
        metadata['slice_thickness'] = ''

    try:
        metadata['series_description'] = img.GetMetaData('0008|103e')
    except:
        metadata['series_description'] = ''

    try:
        metadata['patient_id'] = img.GetMetaData('0010|0020')
    except:
        metadata['patient_id'] = ''

    try:
        metadata['age'] = int(re.sub('\D', '', img.GetMetaData('0010|1010')))
    except:
        metadata['age'] = ''

    try:
        metadata['weight'] = float(img.GetMetaData('0010|1030'))
    except:
        metadata['weight'] = ''

    try:
        metadata['size'] = float(img.GetMetaData('0010|1020'))
        if metadata['size'] > 5:
            metadata['size'] /= 100.0
    except:
        metadata['size'] = ''

    return vol, metadata


def load_rbv(path):

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)

    rbv = np.asarray(map(lambda d: sitk.GetArrayFromImage(sitk.ReadImage(d))[0], dicom_names))
    rbv = sitk.GetImageFromArray(rbv)

    return rbv


def load_serie_2(path, slice_by_slice=False):
    metadata = {}

    reader = sitk.ImageSeriesReader()
    reader.SetOutputPixelType(sitk.sitkFloat32)
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)

    img = sitk.ReadImage(dicom_names[0])

    get_series_tags(dicom_series=img, metadata=metadata)
    get_patient_tags(dicom_series=img, metadata=metadata, path=path)

    if not slice_by_slice:
        vol = reader.Execute()
    else:
        vol = [sitk.GetArrayFromImage(sitk.ReadImage(d))[0] for d in dicom_names]
        vol = np.asarray(vol)
        vol = sitk.GetImageFromArray(vol)
        get_instances_tags(dicom_names=dicom_names, metadata=metadata)
    
    # dicom_metadata = read_file(dicom_names[0])
    # del dicom_metadata[Tag(['6000','3000'])] #Del the pixel array data from metadata

    # metadata['dicom_metadata'] = dicom_metadata

    try:
        metadata['patients_name'] = img.GetMetaData('0010|0010')
    except:
        metadata['patients_name'] = ''
  
    return vol, metadata


def get_series_tags(dicom_series, metadata):
    metadata['SOP_class_uid'] = dicom_series.GetMetaData('0008|0016')
    metadata['SOP_instance_uid'] = dicom_series.GetMetaData('0008|0018')
    metadata['modality'] = dicom_series.GetMetaData('0008|0060')
    metadata['study_instance_uid'] = dicom_series.GetMetaData('0020|000d')
    metadata['pixel_spacing'] = [str(dicom_series.GetSpacing()[0]), str(dicom_series.GetSpacing()[1])]
    metadata['spacing_between_slices'] = str(dicom_series.GetSpacing()[2])
    metadata['serie_number'] = dicom_series.GetMetaData('0020|0011')
    metadata['instance_number'] = dicom_series.GetMetaData('0020|0013')
    metadata['acquisition_time'] = dicom_series.GetMetaData('0018|1060')
    metadata['patient_position'] = dicom_series.GetMetaData('0020|0032')
    try:
        metadata['slice_thickness'] = dicom_series.GetMetaData('0018|0050')
    except:
        metadata['slice_thickness'] = None

    try:
        metadata['series_description'] = dicom_series.GetMetaData('0008|103e')
    except:
        metadata['series_description'] = None

    try:
        metadata['heart_rate'] = int(dicom_series.GetMetaData('0018|1088'))
    except:
        metadata['heart_rate'] = None

    try:
        metadata['flip_angle'] = int(dicom_series.GetMetaData('0018|1314'))
    except:
        metadata['flip_angle'] = None

    try:
        metadata['R-R'] = int(dicom_series.GetMetaData('0018|1062'))
    except:
        metadata['R-R'] = None      

def get_patient_tags(dicom_series, metadata, path=None):
    metadata['patient_birth_date'] = dicom_series.GetMetaData('0010|0030')
    try:
        metadata['patient_orientation'] = dicom_series.GetMetaData('0020|0037').split('\\')
    except:
        metadata['patient_orientation'] = None
        ADC_REGEX = '(ADC|adc)'
        parent_dir = os.path.abspath(os.path.join(path, os.pardir))
        for serie in os.listdir(parent_dir):
            if re.search(ADC_REGEX, serie):
                bval_path = glob(parent_dir + '/BVAL*/*')[0]
                bval = sitk.ReadImage(bval_path)
                metadata['patient_orientation'] = bval.GetMetaData('0020|0037').split('\\')

    try:
        metadata['patient_id'] = dicom_series.GetMetaData('0010|0020')
    except:
        metadata['patient_id'] = None

    try:
        metadata['age'] = int(re.sub('\D', '', dicom_series.GetMetaData('0010|1010')))
    except:
        metadata['age'] = None

    try:
        metadata['weight'] = float(dicom_series.GetMetaData('0010|1030'))
    except:
        metadata['weight'] = None

    try:
        metadata['size'] = float(dicom_series.GetMetaData('0010|1020'))
        if metadata['size'] > 5:
            metadata['size'] /= 100.0
    except:
        metadata['size'] = None


def get_instances_tags(dicom_names, metadata):
    ti = [int(round(read_file(d)[0x18, 0x82].value)) for d in dicom_names]
    slice_loc = [float(read_file(d)[0x20, 0x1041].value) for d in dicom_names]
    trigger_time = [int(read_file(d)[0x18, 0x1060].value) for d in dicom_names]
    rr = int(read_file(dicom_names[0])[0x18, 0x1062].value)

    
    metadata['TI'] = ti
    metadata['slice_location'] = slice_loc
    metadata['trigger_time'] = trigger_time
    metadata['R-R'] = rr


def bval_for_all(raw_path):
    patients = glob(raw_path + '/*')

    for patient in patients:
        try:
            bval_from_dwi(patient)
            print 'Done it for ' + patient.split('/')[-1]

        except:
            print 'Got wrong for ' + patient.split('/')[-1]


def bval_from_dwi(patient_path):
    dwis = glob(patient_path + '/*/DWI_FOCUS*')
    # dwis = glob(patient_path + '/DWI_FOCUS*')

    if not dwis:
        raise Exception('This patient path is not valid or it has not a valid DWI folder')

    for dwi in dwis:

        new_folder = glob(patient_path + '/*')[0] + '/BVAL' + dwi.split('DWI_FOCUS')[-1]
        # new_folder = patient_path + '/BVAL' + dwi.split('DWI_FOCUS')[-1]

        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)
        elif glob(new_folder + '/*'):
            raise Exception('There is already a folder called BVAL_' + dwi.split('DWI_FOCUS')[-1] + ' for this patient')

        total = len(glob(dwi + '/*'))
        metade = int(total / float(2))

        for i in range(metade):
            aux = metade + i + 1

            dicom_name = glob(dwi + '/*' + str(aux) + '.dcm')[0]

            file_name = dicom_name.split('/')[-1]

            shutil.copy(dicom_name, new_folder + '/' + file_name)


def get_min_dim(img_arr):
    dim_arr = []
    for img in img_arr:
        dim1 = img.shape[0]
        dim2 = img.shape[1]
        dim_arr.append(dim1)
        dim_arr.append(dim2)
    return min(dim_arr)


def resize_series(serie, rows, cols):
    resize_serie = []
    for img in serie:
        resize_serie.append(imresize(img, (rows, cols)))
    return resize_serie


def check_series_depth(*args):

    img_len_arr = []
    for arg in args:
        img_len_arr.append(len(arg))

    if len(set(img_len_arr)) > 1:
        print 'Depth values: ', set(img_len_arr)
        raise Exception('Error: Different depths from the image series')

    return True, list(set(img_len_arr))[0]


def save_serie(serie, np_label=None, path='../../data/processed/',
               filename=None, extension='dcm', metadata={}, affine=np.eye(4)):
    '''
    Method that saves a 3D image data. Optional it can be saved with a label contour overlay.

    Parameters
    ----------
    serie : array, numpy-array or SimpleITK image
        If array - serie list where each element (the serie slice) is a numpy array
        If SimpleITK image - 3D volume, SimpleITK image

        ex: serie=[numpy-array0, numpy-array1, numpy-array2, ...]
            serie=SimpleITK.Image

    serie_type : string
        Specify if the serie input is a numpy array or a SimpleITK image. Two possible entries 'numpy'
        or 'sitk'. Otherwise, it raises an exception.

    np_label : array, array, numpy-array
        Serie list where each element (the serie slice) is a numpy array of the segmentation label.

        ex: np_label=[label-numpy-array0, label-numpy-array1, label-numpy-array2, ...]

    path : string
        Path where file is to be saved. (Default: path='../../data/processed/')

    filename : string
        File name. It's optional. If None, the working directory name will be chosen. (Default: filename=None)

    extension : string
        The extension to save the serie. It accepts any medical image format accepted by SimpleITK.WriteImage().
        (Default: extension='dcm')

    metadata : dict
        DICOM header standard to specify a study

    Returns
    -------

    Nothing (y)

    '''
    DEFAULT_PATH = '../../data/processed/'
    date = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")

    # 0 - Starting time
    start = time()
    print ''
    print 'SAVE SERIE'
    print '-----------------------------------------------------------'

    # 1 - Get hyp name
    work_dir = os.getcwd().split('/')
    hyp_name = list(reversed(work_dir))[0]
    if not filename:
        filename = hyp_name

    # -- Gambs to change minimal things in this method
    if path == DEFAULT_PATH:
        # 2 - Create hyps output directory
        hyp_directory = path + hyp_name
        if not os.path.exists(hyp_directory):
                os.makedirs(hyp_directory)
    else:
        hyp_directory = path
    # -- end gambs

    # 3a - Get series type
    if all(isinstance(s, (np.ndarray, np.generic)) for s in serie):
        serie_type = 'numpy'
    elif type(serie) == sitk.SimpleITK.Image:
        serie_type = 'sitk'

    # 3b - Get sitk from numpy if that is the case
    if serie_type == 'numpy':
        np_serie = serie
        sitk_serie = sitk.GetImageFromArray(np_serie)
    elif serie_type == 'sitk':
        sitk_serie = serie
    else:
        raise Exception('Image serie format not supported')

    # 3c - Save image with overlay
    if np_label:
        if serie_type == 'sitk':
            np_serie = sitk.GetArrayFromImage(sitk_serie)
            # raise Exception('Serie type must be numpy to save with overlay')

        min_dim_serie = get_min_dim(np_serie)
        min_dim_label = get_min_dim(np_label)
        min_dim = min(min_dim_serie, min_dim_label)
        n = np.array(resize_series(np_serie, min_dim, min_dim))
        l = np.array(resize_series(np_label, min_dim, min_dim))
        sitk_serie = sitk.GetImageFromArray(n)
        sitk_label = sitk.GetImageFromArray(l)
        sitk_serie = sitk.LabelOverlay(sitk_serie, sitk.LabelContour(sitk_label))

    # 4 - Write image
    # 4a - if DICOM
    if extension == 'dcm':

        study_directory = os.path.join(hyp_directory, 'study')
        if not os.path.exists(study_directory):
            os.makedirs(study_directory)
        else:
            study_directory = study_directory + '_({})'.format(date)
            os.makedirs(study_directory)

        if not metadata:
            raise Exception('There is not sufficient metadata to create a dicom serie')

        for i in range(sitk_serie.GetSize()[2]):

            metadata['patient_position'] = [str(sitk_serie.TransformIndexToPhysicalPoint([0, 0, i])[0]),
                                            str(sitk_serie.TransformIndexToPhysicalPoint([0, 0, i])[1]),
                                            str(sitk_serie.TransformIndexToPhysicalPoint([0, 0, i])[2])]
            metadata['instance_number'] = i + 1
            metadata['SOP_instance_uid'] = 'vacuo' + str(i + 1)

            write_dicom(sitk.GetArrayFromImage(sitk_serie[:, :, i]), os.path.join(study_directory, '{}_{}.{}'.format(filename, i + 1, extension)), metadata)
        print 'Study directory created... [' + study_directory + ']'

    # 4b - if NIFTI
    elif extension == 'nii':
        file_image_name = os.path.join(hyp_directory, '{}.{}'.format(filename, extension))
        if os.path.exists(file_image_name):
            file_image_name = os.path.join(hyp_directory, '{}'.format(filename)) + '_({}).{}'.format(date, extension)

        np_serie = sitk.GetArrayFromImage(serie)
        np_serie = np.swapaxes(np.swapaxes(np_serie, 0, 1), 1, 2)  # Changing the format from (Z,X,Y) -> (X,Y,Z)
        nii_serie = Nifti1Image(np_serie, affine)
        save(nii_serie, file_image_name)

    # 4c - if OTHER EXTENSIONS
    else:
        file_image_name = os.path.join(hyp_directory, '{}.{}'.format(filename, extension))
        if os.path.exists(file_image_name):
            file_image_name = os.path.join(hyp_directory, '{}'.format(filename)) + '_({}).{}'.format(date, extension)
        sitk.WriteImage(sitk_serie, file_image_name)
        print 'File saved... [' + file_image_name + ']'

    # 5 - Time elapsed
    time_elapsed = time() - start
    time_units = 'seconds'
    if time_elapsed > 60:
        time_elapsed /= 60
        time_units = 'minutes'

    # 6 - Final Prints
    print '> Time elapsed:', time_elapsed, time_units
    print '-----------------------------------------------------------'

    return


def write_dicom(pixel_array, filename, metadata):
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = metadata['SOP_class_uid']
    file_meta.MediaStorageSOPInstanceUID = metadata['SOP_instance_uid']
    file_meta.ImplementationClassUID = 'vacuo'

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble="\0" * 128)

    ds.Modality = 'MR'
    ds.ContentDate = str(date.today()).replace('-', '')
    ds.ContentTime = str(time())  # milliseconds since the epoch

    ds.SeriesNumber = metadata['serie_number']
    ds.SeriesDescription = metadata['series_description'] + '_T1_map'
    ds.InstanceNumber = metadata['instance_number']
    ds.StudyInstanceUID = metadata['study_instance_uid']
    ds.PatientID = metadata['patient_id']
    ds.PatientsName = metadata['patients_name']
    ds.PatientBirthDate = metadata['patient_birth_date']
    ds.PixelSpacing = metadata['pixel_spacing']
    ds.SpacingBetweenSlices = metadata['spacing_between_slices']
    ds.SliceThickness = metadata['slice_thickness']
    ds.ImageOrientationPatient = metadata['patient_orientation']
    ds.SOPClassUID = metadata['SOP_class_uid']
    ds.SOPInstanceUID = metadata['SOP_instance_uid']
    ds.ImagePositionPatient = metadata['patient_position']

    # These are the necessary imaging components of the FileDataset object.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    # ds.SmallestImagePixelValue = '\\x00\\x00'
    # ds.LargestImagePixelValue = '\\xff\\xff'
    ds.Rows = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]

    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()

    ds.save_as(filename)


def sitk2np(array_sitk):
    """Transform an array of sitk image to an array of np in place."""
    array_np = []
    for i in range(len(array_sitk)):
        array_np.append(sitk.GetArrayFromImage(array_sitk[i]))

    return array_np


def show_random_train(X, y, number_studies=3, number_slices=1, hist=True):
    """
    Show random training examples.

    X: list of studies
    y: target
    number_studies: number of studies to be shown
    number_slices: slices to be shown for each study
    hist: show histogram
    """
    assert len(X) == len(y), \
        "length of train (%i) is not the same of label (%i)" % (len(X), len(y))
    assert len(X) > number_studies, \
        "number of studies to show (%i) is bigger than the length of train (%i)" % (number_studies, len(X))

    indices = random.sample(range(len(X)), number_studies)
    for i in indices:
        # assert X[i].shape == y[i].shape, \
        #     "shape from X and y not matching"
        if X[i].shape[0] < number_slices:
            print("Not enough slices for this study... Printing all of them")
            number_slices = X[i].shape[0]
        slices = random.sample(range(X[i].shape[0]), number_slices)
        print "Printing study number", str(i)
        for j in slices:
            plt.subplot(221)
            plt.imshow(X[i][j], cmap='Greys_r')
            plt.subplot(222)
            plt.imshow(y[i][j], cmap='Greys_r')
            if hist:
                plt.subplot(223)
                plt.hist(X[i][j])
                plt.subplot(224)
                plt.hist(y[i][j])
            plt.show()


def export_volumetric_images(img, img_name, pattern='{i}_{img}'):
    # I will use skimage since I do not have openCV here
    from skimage import io

    for i, im in enumerate(img):
        io.imsave('{}_{}'.format(i=i, img=img_name) + '.png', im)


def split_examples(X, y, random_state=12, test_size=0.3, n_splits=1):
    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    try:
        example_split = X.shape[0]
    except:
        example_split = len(X)

    for train_index, test_index in rs.split(range(example_split)):
        X_train, X_valid = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_valid = [y[i] for i in train_index], [y[i] for i in test_index]

        yield {
            'X_train': np.array(list(chain(*X_train))),
            'X_valid': np.array(list(chain(*X_valid))),
            'y_train': np.array(list(chain(*y_train))),
            'y_valid': np.array(list(chain(*y_valid))),
        }


def cross_validation_validated(X, y, n_splits=10, random_state=42):
    '''
    Return the indices for cross valdiation for each image
    X: train data should be like:
    [
        [slice1, slice2, ...],  # example 1
        [slice1, slice2, ...],  # example 2
        .
        .
        .
    ]
    y: array-like

    return an iterator containing dictionaries of train and valid data and target
    '''
    try:
        example_split = X.shape[0]
    except:
        example_split = len(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # print('*' * 80)
    # print np.array(X[0]).shape, np.array(y[0]).shape
    # print('*' * 80)

    for train_index, test_index in kf.split(range(example_split)):
        X_train, X_valid = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_valid = [y[i] for i in train_index], [y[i] for i in test_index]

        print len(list(chain(*X_train))), len(list(chain(*y_train))), np.array(X_train[0]).shape, np.array(y_train[0]).shape
        # raise Exception('vaca')

        yield {
            'X_train': list(chain(*X_train)),
            'X_valid': list(chain(*X_valid)),
            'y_train': list(chain(*y_train)),
            'y_valid': list(chain(*y_valid)),
        }


def read_image(img_filepath):
    return imread(img_filepath, as_grey=True)

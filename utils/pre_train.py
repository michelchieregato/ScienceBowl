# coding: utf-8
from itertools import chain, combinations
import numpy as np
import SimpleITK as sitk
from skimage.transform import rotate
from random import shuffle, seed
from tqdm import tqdm
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
try:
    from sklearn.model_selection import KFold, ShuffleSplit
except ImportError:
    from sklearn.cross_validation import KFold, ShuffleSplit


SEED = 42


def data_augmentation(X, y, normalization=True, rotation_range=0., translation_range=(0., 0.),
                      zoom_range=0., horizontal_flip=False, vertical_flip=False,
                      rescale=None, yield_size=32):
    '''
    Method that wraps the preprocessing methods from Keras.
    It can be mainly used for data normalization and data augumentation.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features

    y : array-like, shape (n_samples,)
        Target values

    normalization : bool
                   Divide inputs by std of the dataset. (Default=True)

    rotation_range : tuple shape-like
                    tuples for degrees (0 to 180) range. (Default=(0.,0.))

    zoom_range : float or scalar
                amount of zoom. if scalar z, zoom will be randomly picked
                in the range [1-z, 1+z]. A sequence of two can be passed instead
                to select this range. (Default=0)

    horizontal_flip : bool
                     whether to randomly flip images horizontally. (Default=False)
    vertical_flip : bool
                    whether to randomly flip images vertically. (Default=False)

    rescale : None or float
              rescaling factor. If None or 0, no rescaling is applied,
              otherwise we multiply the data by the value provided (before applying
              any other transformation).

    yield_size: int
                batches of augmented/normalized data. Yields batches
                indefinitely, in an infinite loop

    Returns
    -------

    Returns a generator that takes numpy data & label arrays, and generates
    batches of augmented/normalized data. Yields batches indefinitely, in an
    infinite loop.

    Notes
    -----

    !!!
    Note that it returns a GENERATOR. So, to use Keras .fit, we must apply
    instead fit_generator:
        model.fit_generator(generator_from_this_function)
    !!!
    '''
    data_generator = ImageDataGenerator(
        featurewise_std_normalization=normalization,
        rotation_range=rotation_range,
        width_shift_range=translation_range[0],
        height_shift_range=translation_range[1],
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=rescale
    )

    data_generator.fit(X)

    return data_generator.flow(X, y, batch_size=yield_size)


def vertical_flip(image, dummy):
    return image[:, ::-1, :]


def horizontal_flip(image, dummy):
    return image[:, :, ::-1]


def rotation_3d(image, angle):

    max_value = np.max(image)
    normalized_image = image / max_value
    rotated_image = []
    for z in normalized_image:
        rotated_image.append(rotate(z, angle))

    rotated_image = np.asarray(rotated_image)
    rotated_image = rotated_image * max_value

    return rotated_image


def data_augmentation_3d(X, y, len_epoch, transformations={'horizontal_flip': True, 'rotation_range': 4}):

    transformation_dict = {
        'horizontal_flip': horizontal_flip,
        'rotation_range': rotation_3d,
        'vertical_flip': vertical_flip,
    }

    assert all(transformation in transformation_dict for transformation in transformations), 'Not all transformations are known.'
    assert len(X) == len(y), 'Train and Target arrays must have the same length.'

    subsets = []
    for n in range(len(transformations)):
        subsets.extend(list(combinations(transformations.keys(), n + 1)))

    max_len_epoch = (len(subsets) + 1) * len(X)
    assert len_epoch <= max_len_epoch, 'len_epoch must be smaller than {} (max_len_epoch)'.format(max_len_epoch)

    n_augumented_data = len_epoch - len(X)

    # create pairs (Img, tranformation)
    xy_pair_vol_transformation_arr = []

    for vol_x, vol_y in zip(X, y):
        for t in subsets:
            xy_pair_vol_transformation_arr.append([vol_x, vol_y, t])

    seed(SEED)
    shuffle(xy_pair_vol_transformation_arr)

    x_aug = list(X)
    y_aug = list(y)

    for xyt in tqdm(xy_pair_vol_transformation_arr[:n_augumented_data], desc='data augumentation'):
        vol_x, vol_y, t = xyt[0], xyt[1], xyt[2]
        for tt in t:
            vol_x = transformation_dict[tt](vol_x, transformations[tt])
            vol_y = transformation_dict[tt](vol_y, transformations[tt])

        x_aug.append(vol_x)
        y_aug.append(vol_y)

    print 'Data Augumentation 3D completed successfully!'

    return x_aug, y_aug


def check_backend():
    '''
    Check if backend and image dimension ordering match, and return their values
    '''
    backend = K.backend()
    image_dim_ordering = K.image_dim_ordering()

    if (backend == 'tensorflow') and (image_dim_ordering == 'th'):
        K.set_image_dim_ordering('tf')
        image_dim_ordering = K.image_dim_ordering()
        print "WARNING: Backend is TensorFlow, image dimension ordering was set to 'tf' to resolve mismatch"
    elif (backend == 'theano') and (image_dim_ordering == 'tf'):
        K.set_image_dim_ordering('th')
        image_dim_ordering = K.image_dim_ordering()
        print "WARNING: Backend is Theano, image dimension ordering was set to 'th' to resolve mismatch"

    return backend, image_dim_ordering


def pre_reshape(series):
    """
    Prepare input for models.

    First, check what backend is used and make sure it matches with dimension ordering.
    Then, reshape lists of image series = ([series0], [series1], ...) where each
    series = (samples, rows, cols) and all images have equal dimensions (output
    of MakeDataSet.resize()) to 4-dimensional arrays = (samples, channels, rows, cols)
    """
    if isinstance(series, list):
        series = np.asarray(series)

    backend, _ = check_backend()

    if backend == 'tensorflow':
        # Shape = (samples, rows, cols, channels) if dim_ordering='tf'
        if series.ndim == 3:
            series = series.reshape(series.shape[0], series.shape[1], series.shape[2], 1)

    elif backend == 'theano':
        # Shape = (samples, channels, rows, cols) if dim_ordering='th'
        if series.ndim == 3:
            series = series.reshape(series.shape[0], 1, series.shape[1], series.shape[2])

    return series


def sitk2np(sitk_series):
    '''
    Cast SimpleITK image series to lists of numpy arrays
    '''
    np_series = []
    for case in sitk_series:
        case = sitk.GetArrayFromImage(case)
        np_series.append(case)

    return np_series


def split_examples(data, target, random_state=12, test_size=0.3, n_splits=1):
    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    try:
        example_split = data.shape[0]
    except:
        example_split = len(data)

    for train_index, test_index in rs.split(range(example_split)):
        X_train, X_valid = [data[i] for i in train_index], [data[i] for i in test_index]
        y_train, y_valid = [target[i] for i in train_index], [target[i] for i in test_index]

        yield {
            'X_train': np.array(list(chain(*X_train))),
            'X_valid': np.array(list(chain(*X_valid))),
            'y_train': np.array(list(chain(*y_train))),
            'y_valid': np.array(list(chain(*y_valid))),
        }


def cross_validation(data, target, n_splits=10, random_state=42):
    '''
    Split data set in training and validation sets,
    keeping images of the same case in only one of the two sets.

    Return the indices for cross valdiation for each image
    data: train data should be like:
    [
        [slice1, slice2, ...],  # example 1
        [slice1, slice2, ...],  # example 2
        .
        .
        .
    ]
    target: array-like

    return an iterator containing dictionaries of train and valid data and target
    '''
    try:
        example_split = data.shape[0]
    except:
        example_split = len(data)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, test_index in kf.split(range(example_split)):
        X_train, X_valid = [data[i] for i in train_index], [data[i] for i in test_index]
        y_train, y_valid = [target[i] for i in train_index], [target[i] for i in test_index]

        yield {
            'X_train': np.array(list(chain(*X_train))),
            'X_valid': np.array(list(chain(*X_valid))),
            'y_train': np.array(list(chain(*y_train))),
            'y_valid': np.array(list(chain(*y_valid))),
        }

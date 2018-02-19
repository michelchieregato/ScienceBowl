# coding: utf-8
import numpy as np
import SimpleITK as sitk
from scipy.misc import imresize
from scipy.ndimage import generate_binary_structure, binary_closing, binary_opening, label, center_of_mass, binary_fill_holes
from scipy.ndimage.interpolation import zoom
from skimage.exposure import equalize_hist, rescale_intensity
from skimage.filters import threshold_otsu


def binarize(img, threshold=0.5):
    """Turn pixels to 1 if intensity is greater than threshold, 0 otherwise."""
    return np.where(img > threshold, 1, 0)


def normalize(img):
    """Normalize the image i.e. scale all pixels to range 0-1."""
    if img.dtype == np.uint8:
        factor = 255.0
    else:
        factor = np.amax(img)

    return img.astype(np.float32) / factor


def opening(series, n_dim=3, connectivity=3, iterations=2):
    """Morphological opening."""
    '''
    Input is a 3D-image (if n_dim=3) with shape = (samples, rows, cols)
    i.e. an output from the post_reshape() function

    The structuring element (strel) is defined by the image dimension (default is 3D)
    and by the connectivity (default is 3, i.e. a 3x3x3 cube of ones)

    The opening operation is performed for the number of times specified
    in iterations (default is 2)
    '''
    if isinstance(series, list):
        series = np.asarray(series)

    strel = generate_binary_structure(n_dim, connectivity)
    series = binary_opening(series, strel, iterations)

    processed_series = []
    for i in range(series.shape[0]):
        processed_series.append(series[i])

    return processed_series

def fill_holes(series, n_dim=3, connectivity=3, iterations=10):
    
    if isinstance(series, list):
        series = np.asarray(series)

    strel = generate_binary_structure(n_dim, connectivity)
    series = binary_fill_holes(series, strel, iterations)

    processed_series = []
    for i in range(series.shape[0]):
        processed_series.append(series[i])

    return processed_series


def opening_with_strel(series, strel, iterations=2):
    """Morphological opening."""
    '''
    Same as opening, but with a given strel.
    '''
    if isinstance(series, list):
        series = np.asarray(series)

    series = binary_opening(series, strel, iterations)

    processed_series = []
    for i in range(series.shape[0]):
        processed_series.append(series[i])

    return processed_series


def closing(series, n_dim=3, connectivity=3, iterations=2):
    """Morphological closing."""
    '''
    Input is a 3D-image (if n_dim=3) with shape = (samples, rows, cols)
    i.e. an output from the post_reshape() function

    The structuring element (strel) is defined by the image dimension (default is 3D)
    and by the connectivity (default is 3, i.e. a 3x3x3 cube of ones)

    The closing operation is performed for the number of times specified
    in iterations (default is 2)
    '''
    if isinstance(series, list):
        series = np.asarray(series)

    strel = generate_binary_structure(n_dim, connectivity)
    series = binary_closing(series, strel, iterations)

    processed_series = []
    for i in range(series.shape[0]):
        processed_series.append(series[i])

    return processed_series


def crop_resize(img, rows, cols, interp='bilinear'):
    """
    Crop center and resize.

    :param img: image to be cropped and resized.
    """
    img_shape = (rows, cols)
    if img.shape[0] < img.shape[1]:
        img = img.T
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    img = crop_img
    img = imresize(img, img_shape, interp)

    return img / 255.0


def resize_series(series, rows, cols):
    """
    Resize all images in all cases to rows, cols.

    series: array of numpy 3d images
    rows: size of desired row
    cols: size of desired collumms
    TODO: couldn't manage to do this in place, will return to that after the necessity arises
    """
    new_series = list()
    for i in range(len(series)):
        case = np.empty((series[i].shape[0], rows, cols))
        for j in range(series[i].shape[0]):
            case[j] = crop_resize(series[i][j], rows, cols)
        new_series.append(case)
    return new_series


def bounding_box(mask_image):
    """Bounding box."""
    '''
    Calculates the bounding box of a mask image.
    Mask from a non-zero values
    '''
    if np.all(mask_image == False):
        return None

    rows = np.any(mask_image, axis=1)
    cols = np.any(mask_image, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax - cmin, rmax - rmin


def bounding_box_3D(mask_image):
    """Calculate the 3D bounding box from mask image."""
    r = np.any(mask_image, axis=(1, 2))
    c = np.any(mask_image, axis=(0, 2))
    z = np.any(mask_image, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def image_equalization(img, lower_percentile=None, upper_percentile=None):
    """
    Equalize the image and add a threshold for min a lower and upper percentile.

    img: image array
    lower_percentile: lower cut-off
    upper_percentile: upper cut-off
    """
    if lower_percentile:
        p_min = np.percentile(img, lower_percentile)
        img[img < p_min] = p_min

    if upper_percentile:
        p_max = np.percentile(img, upper_percentile)
        img[img > p_max] = p_max

    img = img / np.amax(img)
    return img


def equalize_series_2d(series, lower_percentile=None, upper_percentile=None):
    """
    Equalize series using a threshold for lower and upper percentile.

    series: array of numpy 3d images
    lower_percentile: lower cut-off
    upper_percentile: upper cut-off
    """
    for i in range(len(series)):
        for j in range(series[i].shape[0]):
            series[i][j] = normalize(image_equalization(series[i][j], lower_percentile, upper_percentile))


def calibrate(im):
    new_im = equalize_hist(im)
    img_rescale = rescale_intensity(new_im, out_range=np.uint8)
    new_im = img_rescale.astype(dtype=np.uint8)

    return new_im


def crop_image(img_gray, crop_rate=1.0, min_size=(15, 15), crop_left=0.0, crop_right=0.0, crop_top=0.0, crop_bottom=0.0, center=None):
    h, w = img_gray.shape

    # w2 = max(int(w * crop_rate), min_size[0])
    # h2 = max(int(h * crop_rate), min_size[1])
    w2 = min_size[0]
    h2 = min_size[1]

    if center is None:
        center = (w / 2, h / 2)

    offset_left = int(w2 * crop_left)
    offset_right = int(w2 * crop_right)
    offset_top = int(h2 * crop_top)
    offset_bottom = int(h2 * crop_bottom)

    left = int(center[0] - (w2 / 2))
    right = left + w2
    top = int(center[1] - (h2 / 2))
    bottom = top + h2

    return img_gray[(top + offset_top):(bottom - offset_bottom), (left + offset_left):(right - offset_right)]


def make_relu(bval, adc):
    """Relu implementation for two numpy matrices."""
    if np.any(adc != 0):
        adc_norm = adc / np.amax(adc)
    else:
        adc_norm = adc

    if np.any(bval != 0):
        bval_norm = bval / np.amax(bval)
    else:
        bval_norm = bval

    matriz = bval_norm - adc_norm
    return np.where(matriz < 0, 0, matriz)


def combine_masks(mask1, mask2):
    """Combine mask in one, they should be binary."""
    mask = mask1 + mask2

    return np.where(mask == 2, 1, mask)


def combine_masks_multiclass(mask1, mask2):
    """Combine mask in one multiclass, if there is an intersection, it will replace as it is from the second input mask."""
    mask = mask1 + 2 * mask2

    return np.where(mask == 3, 2, mask)


def compute_3d_otsu_thresh(mask_vol):
    """Compute global otsu for a volume."""
    depth = mask_vol.shape[0]
    
    threshold_global_otsu = 0
    for z in range(depth):
        m = mask_vol[z]
        
        try:
            thresh = threshold_otsu(m)
        except:
            thresh = 0

        if thresh > threshold_global_otsu:
            threshold_global_otsu = thresh

    return threshold_global_otsu


def number_of_centers(mask_vol):
    """Return the quantity of connected elements founded."""
    return label(mask_vol)[1]


def compute_centers_of_mass(volume):
    """TODO."""
    labeled = label(volume)

    cm = {}
    for i in range(labeled[1]):
        cm[i + 1] = center_of_mass(volume, labels=labeled[0], index=i + 1)

    return cm


def compute_median_serie(serie):

    if not isinstance(serie, sitk.SimpleITK.Image) and isinstance(serie, numpy.ndarray):
        serie = sitk.GetImageFromArray(serie)

    brain_length = int(img.GetMetaData('0020|1002'))
    n_brains = int(img.GetMetaData('0020|0105'))

    vols = np.asarray(map(lambda i: sitk.GetArrayFromImage(serie[:,:,i:(i + brain_length)]), brain_length*np.asarray(range(n_brains))))
    vols = np.median(vols, axis=0)
    median_serie = sitk.GetImageFromArray(vols)
    median_serie.SetOrigin(serie[:,:,0:brain_length].GetOrigin())
    median_serie.SetSpacing(serie[:,:,0:brain_length].GetSpacing())
    median_serie.SetDirection(serie[:,:,0:brain_length].GetDirection())

    return median_serie


def resample(sitk_image, new_spacing=[1, 1, 1], mask=None):

    # === new_spacing = [z, x, y] ===

    # Determine current pixel spacing
    spacing = np.array([sitk_image.GetSpacing()[2], sitk_image.GetSpacing()[0], sitk_image.GetSpacing()[1]])
    image = sitk.GetArrayFromImage(sitk_image)

    # Calculate resize factor
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    # Resize image
    min_value = np.min(image)
    image = zoom(image, real_resize_factor)
    image = np.where(image < min_value, min_value, image)

    # Return image to sitk image
    resampled_sitk_image = sitk.GetImageFromArray(image)
    new_spacing_sitk = [new_spacing[1], new_spacing[2], new_spacing[0]]
    resampled_sitk_image.SetSpacing(new_spacing_sitk)

    if (isinstance(mask, list) or isinstance(mask, (np.ndarray, np.generic))) and (len(mask) > 0):
        assert mask.ndim == 3, 'Mask dimension must be eqaul to 3 (Current value = {})'.format(mask.ndim)
        mask = zoom(mask, real_resize_factor, mode='nearest')
        mask = np.where(mask > 0.5, 1, 0)

    return resampled_sitk_image, mask

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compute_difference(ADC, BVAL):
    assert (len(ADC) == len(BVAL)), "Images must have the same shape:"
    error = 0
    for i in range(len(ADC)):
        error = error + mse(ADC[i], BVAL[i])
    return error

def substitute_negative_values(t, mean):
    if (t < 0):
        t = mean
    return t

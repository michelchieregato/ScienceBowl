import numpy as np
from matplotlib import pyplot as plt
from medpy.metric.binary import hd
from sklearn import metrics
from keras import backend as K
from image_io import check_series_depth
from SimpleITK import HausdorffDistanceImageFilter, GetImageFromArray


# ============================== METRICS FOR MODEL EVALUATION ===========================

def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    smooth = 0.01
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return (2. * intersection.sum() + smooth) / (im1.sum() + im2.sum() + smooth)


def recall_eval(y_true, y_pred, recall_avarege='binary', threshold=0.5):

    if not isinstance(y_true, (np.ndarray, np.generic)):
        y_true = np.asarray(y_true)

    if not isinstance(y_pred, (np.ndarray, np.generic)):
        y_pred = np.asarray(y_pred)

    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    y_pred = np.where(np.array(y_pred) > threshold, 1, 0)

    return metrics.recall_score(y_true, y_pred, average=recall_avarege)


def dice_eval(target_serie, pred_serie):
    return dice(target_serie, pred_serie)

# def dice_eval(target_serie, pred_serie, plt_hist=False):
#     dice_arr = []
#     count_null_dice = 0
#     # Check the number of slices from the series
#     _, depth = check_series_depth(target_serie, pred_serie)

#     for i in range(depth):
#         dice_value = dice(target_serie[i], pred_serie[i])
#         if not dice_value:
#             count_null_dice += 1
#         else:
#             dice_arr.append(dice_value)
# #     return dice_arr.mean(), min(dice_arr), max(dice_arr) # sum(l) / float(len(l))
#     # Mean calculation without null dice:
#     if plt_hist:
#         plt.hist(dice_arr, bins=50)

#     try:
#         return np.median(dice_arr), np.min(dice_arr), np.max(dice_arr), count_null_dice
#     except:
#         return 0, 0, 0, count_null_dice


# def haussdorf(y_true, y_pred):

#     # haus = HausdorffDistanceImageFilter()
#     # d = haus.Execute(GetImageFromArray(y_true), GetImageFromArray(y_pred))

#     return hausdorff_dist(y_true.flatten(2), y_pred.flatten(2))

def hausdorff_eval(y_true, y_pred, voxelspacing=None, connectivity=1, threshold=0.5):

    if not isinstance(y_true, (np.ndarray, np.generic)):
        y_true = np.asarray(y_true)

    if not isinstance(y_pred, (np.ndarray, np.generic)):
        y_pred = np.asarray(y_pred)

    y_pred = np.where(np.array(y_pred) > threshold, 1, 0)
    try:
        hd_score = hd(y_true, y_pred, voxelspacing=voxelspacing, connectivity=connectivity)
    except:
        hd_score = 0
    return hd_score


def sle(actual, predicted):
    """
    Computes the squared log error.
    This function computes the squared log error between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The squared log error between actual and predicted
    """
    return (np.power(np.log(np.array(actual) + 1) -
            np.log(np.array(predicted) + 1), 2))


def msle(actual, predicted):
    """
    Computes the mean squared log error.
    This function computes the mean squared log error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The mean squared log error between actual and predicted
    """
    return np.mean(sle(actual, predicted))


def rmsle(actual, predicted, *args, **kwargs):
    """
    Computes the root mean squared log error.
    This function computes the root mean squared log error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The root mean squared log error between actual and predicted
    """
    return np.sqrt(msle(actual, predicted))

rmsle_scorer = metrics.scorer.make_scorer(rmsle, greater_is_better=False)


def eval_perf(y_true, y_pred, y_score=None, ids=None, prefix=None):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    print('=== Prediction Results ===')
    results_score = ''

    if ids is not None:
        results_score += 'id,true,pred,proba\n'
        for ID, n_pred, n_true, n_score in zip(ids, y_pred, y_true, y_score):
            results_score += '%s,%d,%d,%.6f\n' % (ID, n_true, n_pred, n_score)
        results_score += '\n'

#     print results_score
#     with open('results/{}-{}.csv'.format(prefix, 'ids_score'), 'w') as f:
#         f.write(results_score)

    # Calculating scores
    score_accuracy = metrics.accuracy_score(y_true, y_pred)
    score_recall = metrics.recall_score(y_true, y_pred, average='weighted')
    score_precision = metrics.precision_score(y_true, y_pred, average='weighted')
    score_f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    score_auc = metrics.roc_auc_score(y_true, y_pred)
    score_rmsle = rmsle(y_true, y_pred)

    results = ''
    # Buffering results
    results += 'Classification results:\n'
    results += 'Accuracy:  %.2f%%\n' % (100 * score_accuracy)
    results += 'Recall:    %.2f%%\n' % (100 * score_recall)
    results += 'Precision: %.2f%%\n' % (100 * score_precision)
    results += 'F1:        %.2f%%\n' % (100 * score_f1)
    results += 'RMSLE:     %.5f\n' % (score_rmsle)
    results += 'ROC AUC:   %.2f%%\n' % (100 * score_auc)
    results += 'Confusion matrix:\n'
    results += '{}\n'.format(metrics.confusion_matrix(y_true, y_pred))
    results += 'Classification report:\n'
    results += '{}\n'.format(metrics.classification_report(y_true, y_pred))

    print(results)

#     if prefix is not None:
#         with open('results/{}-{}.txt'.format(prefix, 'classification_report'), 'w') as f:
#             f.write(results)

    if y_score is not None:
        # ROC CURVE
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure()
        plt.ioff()
        plt.plot(fpr, tpr, lw=1.0, label='ROC (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], '--', lw=0.8, color='0.75')
        plt.plot([0], [0.5], marker='o', markerfacecolor='none', markeredgecolor='red',
                 markersize=8, markeredgewidth=1.0)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid(True)
        # if prefix is not None:
        #     plt.savefig('results/{}-{}.png'.format(prefix, 'roc'), bbox_inches='tight', dpi=300)
        # else:
        plt.show()

    return roc_auc, score_precision, score_recall

# ============================== METRICS FOR MODEL TRAIN ===========================

def dice_coef(y_true, y_pred):
    smooth = 1.

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# def haussdorf(y_true, y_pred):

#     # y_true = K.flatten(y_true)
#     # y_pred = K.flatten(y_pred)

#     # haus = HausdorffDistanceImageFilter()
#     # d = haus.Execute(GetImageFromArray(y_true), GetImageFromArray(y_pred))

#     return HausdorffDist(y_true.flatten(2), y_pred.flatten(2))

# from numpy.core.umath_tests import inner1d


# def HausdorffDist(A,B):
#     # Hausdorf Distance: Compute the Hausdorff distance between two point
#     # clouds.
#     # Let A and B be subsets of metric space (Z,dZ),
#     # The Hausdorff distance between A and B, denoted by dH(A,B),
#     # is defined by:
#     # dH(A,B) = max(h(A,B),h(B,A)),
#     # where h(A,B) = max(min(d(a,b))
#     # and d(a,b) is a L2 norm
#     # dist_H = hausdorff(A,B)
#     # A: First point sets (MxN, with M observations in N dimension)
#     # B: Second point sets (MxN, with M observations in N dimension)
#     # ** A and B may have different number of rows, but must have the same
#     # number of columns.
#     #
#     # Edward DongBo Cui; Stanford University; 06/17/2014

#     # Find pairwise distance


#     # D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
#     # D_mat = K.sqrt((K.expand_dims(A * A)).T + (B * B) - 2 * (K.dot(A, B.T)))
#     # Find DH
#     # dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
#     # dH_1 = K.max(K.min(D_mat, axis=0))
#     # dH_2 = K.max(K.min(D_mat, axis=1))
#     # dH = max([dH_1, dH_2])
#     # print dH_2
#     return K.dot(A, B)
#     return K.variable(value=np.inner1d(A, B))


min_object_size = 1 

def get_labeled_mask(mask, cutoff=.5):
    """Object segmentation by labeling the mask."""
    mask = mask.reshape(mask.shape[0], mask.shape[1])
    lab_mask = skimage.morphology.label(mask > cutoff) 
    
    # Keep only objects that are large enough.
    (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
    if (mask_sizes < min_object_size).any():
        mask_labels = mask_labels[mask_sizes < min_object_size]
        for n in mask_labels:
            lab_mask[lab_mask == n] = 0
        lab_mask = skimage.morphology.label(lab_mask > cutoff) 
    
    return lab_mask 

def get_iou(y_true_labeled, y_pred_labeled):
    """Compute non-zero intersections over unions."""
    # Array of different objects and occupied area.
    (true_labels, true_areas) = np.unique(y_true_labeled, return_counts=True)
    (pred_labels, pred_areas) = np.unique(y_pred_labeled, return_counts=True)

    # Number of different labels.
    n_true_labels = len(true_labels)
    n_pred_labels = len(pred_labels)

    # Each mask has at least one identified object.
    if (n_true_labels > 1) and (n_pred_labels > 1):
        
        # Compute all intersections between the objects.
        all_intersections = np.zeros((n_true_labels, n_pred_labels))
        for i in range(y_true_labeled.shape[0]):
            for j in range(y_true_labeled.shape[1]):
                m = y_true_labeled[i,j]
                n = y_pred_labeled[i,j]
                all_intersections[m,n] += 1 

        # Assign predicted to true background.
        assigned = [[0,0]]
        tmp = all_intersections.copy()
        tmp[0,:] = -1
        tmp[:,0] = -1

        # Assign predicted to true objects if they have any overlap.
        for i in range(1, np.min([n_true_labels, n_pred_labels])):
            mn = list(np.unravel_index(np.argmax(tmp), (n_true_labels, n_pred_labels)))
            if all_intersections[mn[0], mn[1]] > 0:
                assigned.append(mn)
            tmp[mn[0],:] = -1
            tmp[:,mn[1]] = -1
        assigned = np.array(assigned)

        # Intersections over unions.
        intersection = np.array([all_intersections[m,n] for m,n in assigned])
        union = np.array([(true_areas[m] + pred_areas[n] - all_intersections[m,n]) 
                           for m,n in assigned])
        iou = intersection / union

        # Remove background.
        iou = iou[1:]
        assigned = assigned[1:]
        true_labels = true_labels[1:]
        pred_labels = pred_labels[1:]

        # Labels that are not assigned.
        true_not_assigned = np.setdiff1d(true_labels, assigned[:,0])
        pred_not_assigned = np.setdiff1d(pred_labels, assigned[:,1])
        
    else:
        # in case that no object is identified in one of the masks
        iou = np.array([])
        assigned = np.array([])
        true_labels = true_labels[1:]
        pred_labels = pred_labels[1:]
        true_not_assigned = true_labels
        pred_not_assigned = pred_labels
        
    # Returning parameters.
    params = {'iou': iou, 'assigned': assigned, 'true_not_assigned': true_not_assigned,
             'pred_not_assigned': pred_not_assigned, 'true_labels': true_labels,
             'pred_labels': pred_labels}
    return params

def get_score_summary(y_true, y_pred):
    """Compute the score for a single sample including a detailed summary."""
    
    y_true_labeled = get_labeled_mask(y_true)  
    y_pred_labeled = get_labeled_mask(y_pred)  
    
    params = get_iou(y_true_labeled, y_pred_labeled)
    iou = params['iou']
    assigned = params['assigned']
    true_not_assigned = params['true_not_assigned']
    pred_not_assigned = params['pred_not_assigned']
    true_labels = params['true_labels']
    pred_labels = params['pred_labels']
    n_true_labels = len(true_labels)
    n_pred_labels = len(pred_labels)

    summary = []
    for i,threshold in enumerate(np.arange(0.5, 1.0, 0.05)):
        tp = np.sum(iou > threshold)
        fn = n_true_labels - tp
        fp = n_pred_labels - tp
        if (tp+fp+fn)>0: 
            prec = tp/(tp+fp+fn)
        else: 
            prec = 0
        summary.append([threshold, prec, tp, fp, fn])

    summary = np.array(summary)
    score = np.mean(summary[:,1]) # Final score.
    params_dict = {'summary': summary, 'iou': iou, 'assigned': assigned, 
                   'true_not_assigned': true_not_assigned, 
                   'pred_not_assigned': pred_not_assigned, 'true_labels': true_labels,
                   'pred_labels': pred_labels, 'y_true_labeled': y_true_labeled,
                   'y_pred_labeled': y_pred_labeled}
    
    return score, params_dict

def get_score(y_true, y_pred):
    """Compute the score for a batch of samples."""
    scores = []
    for i in tqdm_notebook(range(len(y_true))):
        score,_ = get_score_summary(y_true[i], y_pred[i])
        scores.append(score)
    return np.array(scores)


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
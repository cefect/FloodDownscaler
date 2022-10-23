'''
Created on Sep. 8, 2022

@author: cefect
'''
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
def get_confusion(true_ar, pred_ar, names=['true', 'pred'], method='cat'):
    """get a dataframe confusion matrix for isnull between two arrays"""
    assert isinstance(true_ar, np.ndarray)
    assert true_ar.shape==pred_ar.shape
    #is null confusion (i.e., is dry)
    labels = [True, False]
    if method=='null':
        
        cm_ar = confusion_matrix(np.isnan(true_ar).ravel(), np.isnan(pred_ar).ravel(), labels=labels)
    elif method=='cat':
        cm_ar = confusion_matrix(true_ar.ravel(), pred_ar.ravel(), labels=labels)
    else:
        raise IOError(method)
        
    cm_dx = pd.DataFrame(cm_ar, index=labels, columns=labels).unstack().rename('counts').to_frame()
    cm_dx.index = cm_dx.index.set_names(names).swaplevel()
    cm_dx['codes'] = ['TP', 'FP', 'FN', 'TN']
    return cm_dx.reset_index()
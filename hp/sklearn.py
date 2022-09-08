'''
Created on Sep. 8, 2022

@author: cefect
'''
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
def get_null_confusion(true_ar, pred_ar, names=['true', 'pred']):
    """get a dataframe confusion matrix for isnull between two arrays"""
    assert isinstance(true_ar, np.ndarray)
    assert true_ar.shape==pred_ar.shape
    #is null confusion (i.e., is dry)
    labels = [True, False]
    cm_ar = confusion_matrix(np.isnan(true_ar).ravel(), np.isnan(pred_ar).ravel(), labels=labels)
    cm_dx = pd.DataFrame(cm_ar, index=labels, columns=labels).unstack().rename('counts').to_frame()
    cm_dx.index = cm_dx.index.set_names(names).swaplevel()
    cm_dx['codes'] = ['TP', 'FP', 'FN', 'TN']
    return cm_dx.reset_index()
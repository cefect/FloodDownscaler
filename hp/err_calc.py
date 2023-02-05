'''
Created on Apr. 15, 2022

@author: cefect

tools for calculating model errors
'''
#===============================================================================
# imports
#===============================================================================
import os, sys, datetime, gc, copy,  math, pickle,shutil
 
import pandas as pd
import numpy as np
import scipy.stats

from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ErrorCalcs(object):
    def __init__(self,
            pred_ser=None,
            true_ser=None,
            logger=None,
            normed=False,
            ):
        #attach
        
        self.pred_ser=pred_ser.rename('pred')
        self.true_ser=true_ser.rename('true')
        self.normed=normed
        
        self.check_match()
        
        self.df_raw = pred_ser.rename('pred').to_frame().join(true_ser.rename('true'))
        
        
        self.logger=logger
        
        self.res_d = dict()
        
        self.logger.info(f'on {len(pred_ser)}')
        
        self.data_retrieve_hndls = {
            'bias':         lambda **kwargs:self.get_bias(**kwargs),
            'bias_shift':       lambda **kwargs:self.get_bias1(**kwargs),
            'meanError':    lambda **kwargs:self.get_meanError(**kwargs),
            'meanErrorAbs': lambda **kwargs:self.get_meanErrorAbs(**kwargs),
            'RMSE':         lambda **kwargs:self.get_RMSE(**kwargs),
            'pearson':      lambda **kwargs:self.get_pearson(**kwargs),
            'confusion':    lambda **kwargs:self.get_confusion(**kwargs),
            'stats':        lambda **kwargs:self.get_stats(**kwargs),
            }
        
    def retrieve(self, #skinny retrival
                 dkey,
 
                 logger=None,
                 **kwargs
                 ):
        """based on oop.Session.retrieve"""
 
        if logger is None: logger=self.logger
        log = logger.getChild('ret')
        
        drh_d = self.data_retrieve_hndls
 
        
        assert dkey in drh_d, dkey
        
        f = drh_d[dkey]
        
        return f(dkey=dkey, logger=log, **kwargs)
    
    def get_bias1(self, #shift bias to be zero centered
                  dkey='bias_1',
                  **kwargs):
        return self.get_bias(**kwargs)-1
        
    def get_bias(self,
                 per_element=False,
                 dkey='bias', logger=None,
                 ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        assert dkey=='bias'
        log = logger.getChild('bias')
        pred_ser=self.pred_ser
        true_ser=self.true_ser
        df = self.df_raw.copy()

        
        if not per_element:
            s1 = df.sum()
            return s1['pred']/s1['true']
        else:
            res1 = pred_ser/true_ser
            res1 = res1.rename('bias')
            #=======================================================================
            # both zeros
            #=======================================================================
            #true_bx = true_ser==0
            
            bx = np.logical_and(
                true_ser==0,
                pred_ser==0)
            
            if bx.any():
                log.info('replacing %i/%i zero matches w/ 1.0'%(bx.sum(), len(bx)))
                res1.loc[bx] = 1.0
                
            #=======================================================================
            # pred zeros
            #=======================================================================
            bx = np.logical_and(
                true_ser!=0,
                pred_ser==0)
            
            if bx.any():
                log.info('replacing %i/%i zero mismatches w/ null'%(bx.sum(), len(bx)))
                res1.loc[bx] = np.nan
                
            #=======================================================================
            # true zeros
            #=======================================================================
            bx = np.logical_and(
                true_ser==0,
                pred_ser!=0)
            if bx.any():
                log.info('replacing %i/%i zero mismatches w/ null'%(bx.sum(), len(bx)))
                res1.loc[bx] = np.nan
                
            #=======================================================================
            # wrap
            #=======================================================================
            log.info('finished w/ mean bias = %.2f (%i/%i nulls)'%(
                res1.dropna().mean(), res1.isna().sum(), len(res1)))
            
            """
            bx = res1==np.inf
            
            view(df.join(res1).loc[bx, :])
            res1[bx]
            pred_ser.to_frame().join(true_ser).loc[bx, :]
            """
            
            return res1
    
    def get_meanError(self,
                      dkey='meanError',
                      logger=None
                      ):
        assert dkey=='meanError'
 

        df = self.df_raw.copy()
        
        return (df['pred'] - df['true']).sum()/len(df)
        
    def get_meanErrorAbs(self,
                       dkey='meanErrorAbs',
                      logger=None
                      ):
 
        assert dkey=='meanErrorAbs'
        df = self.df_raw.copy()
        
        return (df['pred'] - df['true']).abs().sum()/len(df)
    
    def get_RMSE(self,
                 dkey='RMSE',
                 logger=None):
        assert dkey=='RMSE'
        df = self.df_raw.copy()
        
        
        return math.sqrt(np.square(df['pred'] - df['true']).mean())
    
    
    def get_all(self, #load all the stats in the retrieve handles 
                dkeys_l = None,
                logger=None):
        
        if dkeys_l is None:
            dkeys_l = self.data_retrieve_hndls.keys()
        
        
        res_d = dict()
        for dkey in dkeys_l:
            res_d[dkey] = self.retrieve(dkey, logger=logger)
            
        return res_d
    
    def get_pearson(self,
                    dkey='pearson',
                    logger=None):
        assert dkey=='pearson'
        df = self.df_raw.copy()
        pearson, pval = scipy.stats.pearsonr(df['true'], df['pred'])
        return pearson
    
    def get_confusion(self,
                      dkey='confusion',
                     wetdry=False,
                     normed=None, #normalize confusion values by total count
                     labels=[True, False],
                     logger=None):
        """get a confusion matrix with nice labels
        
        
        Parmaeters
        ------------
        wetdry: bool
            treat 0 as negative and >0 as positive
        
        Returns
        -------------
        pd.DataFrame
            classic confusion matrix
            
        pd.DataFrame
            unstacked confusion matrix
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_confusion')
        assert dkey=='confusion'
        if normed is None: normed=self.normed
        df_raw = self.df_raw.copy()
        
        #=======================================================================
        # prep data
        #=======================================================================
        #convert to boolean
        if wetdry:
            assert np.array([['float' in e for e in [d.name for d in df_raw.dtypes]]]).all()
            
            df1 = df_raw>0.0
            
            #===================================================================
            # df1 = pd.DataFrame('dry', index=df_raw.index, columns=df_raw.columns)
            # 
            # df1[df_raw>0.0] = 'wet'
            #===================================================================
            
            #labels = ['wet', 'dry']
 
            
        #=======================================================================
        # else:
        #     raise IOError('not impelemented')
        #     df1 = df_raw.copy()
        #     
        #     labels=['pred', 'true']
        #=======================================================================
        assert (df1.dtypes=='bool').all()
 
        #build matrix
        cm_ar = confusion_matrix(df1['true'], df1['pred'], labels=labels)
        
        cm_df = pd.DataFrame(cm_ar, index=labels, columns=labels)
        
        #=======================================================================
        # normalize
        #=======================================================================
        if normed:
            cm_df = cm_df/len(df_raw)
        
        #convert and label
        
        cm_df2 = cm_df.unstack().rename('counts').to_frame()
        
        cm_df2.index.set_names(['true', 'pred'], inplace=True)
        
        cm_df2['codes'] = ['TP', 'FP', 'FN', 'TN']
        
        cm_df2 = cm_df2.set_index('codes', append=True)
        
        return cm_df, cm_df2.swaplevel(i=0, j=2)
    
    def get_stats(self, #get baskc stats on one series
                  ser=None,
                  logger=None,
                  dkey='stats',
                  stats_l = ['min', 'mean', 'max'],
                  ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_stats')
        assert dkey=='stats'
        if ser is None: ser = self.pred_ser
        
        return {stat:getattr(ser, stat)() for stat in stats_l}
        
    
    def check_match(self,
                    pred_ser=None,
                    true_ser=None,
                    ):
        if pred_ser is None:
            pred_ser=self.pred_ser
        if true_ser is None:
            true_ser=self.true_ser
            
        assert isinstance(pred_ser, pd.Series)
        assert isinstance(true_ser, pd.Series)
        
        assert_index_equal(pred_ser.index, true_ser.index)
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args,**kwargs):
        for k in copy.copy(list(self.__dict__.keys())):
            del self.__dict__[k]
 

def get_confusion_cat(true_arB, pred_arB, 
                      confusion_codes={'TP':'TP', 'TN':'TN', 'FP':'FP', 'FN':'FN'},
                      ):
    """compute the confusion code for each element
    
    Parameters
    -----------
    confusion_codes: dict
        optional mapping for naming the 4 confusion categories
    """
    #start with dummy
    res_ar = np.full(true_arB.shape, np.nan)
    #true positives
    res_ar = np.where(
        np.logical_and(true_arB, pred_arB), 
        confusion_codes['TP'], res_ar)
    #true negatives
    res_ar = np.where(
        np.logical_and(np.invert(true_arB), np.invert(pred_arB)), 
        confusion_codes['TN'], res_ar)
    #false positives
    res_ar = np.where(
        np.logical_and(np.invert(true_arB), pred_arB), 
        confusion_codes['FP'], res_ar)
    #false negatives
    res_ar = np.where(
        np.logical_and(true_arB, np.invert(pred_arB)), 
        confusion_codes['FN'], res_ar)
    
    
    return res_ar        
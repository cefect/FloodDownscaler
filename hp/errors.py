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


from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ErrorCalcs(object):
    def __init__(self,
            pred_ser=None,
            true_ser=None,
            logger=None,
            ):
        #attach
        
        self.pred_ser=pred_ser.rename('pred')
        self.true_ser=true_ser.rename('true')
        
        
        self.check_match()
        
        self.df_raw = pred_ser.rename('pred').to_frame().join(true_ser.rename('true'))
        
        
        self.logger=logger
        
        self.res_d = dict()
        
        
        self.data_retrieve_hndls = {
            'bias':         lambda **kwargs:self.get_bias(**kwargs),
            'meanError':    lambda **kwargs:self.get_meanError(**kwargs),
            'meanErrorAbs': lambda **kwargs:self.get_meanErrorAbs(**kwargs),
            'RMSE':         lambda **kwargs:self.get_RMSE(**kwargs),
            'confusion':    lambda **kwargs:self.get_confusion(**kwargs),
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

        start = datetime.datetime.now()
        
        assert dkey in drh_d
        
        f = drh_d[dkey]
        
        return f(dkey=dkey, logger=log, **kwargs)
        
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
        return math.sqrt(((df['pred'] - df['true'])**2).sum()/len(df))
    
    
    def get_all(self, #load all the stats in the retrieve handles 
                logger=None):
        
        
        res_d = dict()
        for dkey in self.data_retrieve_hndls.keys():
            res_d[dkey] = self.retrieve(dkey, logger=logger)
            
        return res_d
    
    def get_confusion(self,
                      dkey='confusion',
                     wetdry=True,
                     logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_confusion')
        assert dkey=='confusion'
        
        df_raw = self.df_raw.copy()
        
        #=======================================================================
        # prep data
        #=======================================================================
        if wetdry:
            assert (df_raw.dtypes == 'float64').all()
            
            df1 = pd.DataFrame('dry', index=df_raw.index, columns=df_raw.columns)
            
            df1[df_raw>0.0] = 'wet'
            
            labels = ['wet', 'dry']
            
        else:
            raise IOError('not impelemented')
            df1 = df_raw.copy()
            
 
        #build matrix
        cm_ar = confusion_matrix(df1['true'], df1['pred'], labels=labels)
        
        cm_df = pd.DataFrame(cm_ar, index=labels, columns=labels)
        
        #convert and label
        
        cm_df2 = cm_df.unstack().rename('counts').to_frame()
        
        cm_df2.index.set_names(['true', 'pred'], inplace=True)
        
        cm_df2['codes'] = ['TP', 'FP', 'FN', 'TN']
        
        cm_df2 = cm_df2.set_index('codes', append=True)
        
        return cm_df, cm_df2.swaplevel(i=0, j=2)
        
    
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
        
        
        
        
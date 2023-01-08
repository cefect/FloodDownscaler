'''
Created on Jan. 6, 2023

@author: cefect

validating downscaling results
'''
import logging, os, copy, datetime
import numpy as np
import pandas as pd
import rasterio as rio
from sklearn.metrics import confusion_matrix

from hp.rio import (
    RioSession, RioWrkr, assert_rlay_simple, get_stats, assert_spatial_equal,
    )

from hp.logr import get_new_console_logger
from fdsc.scripts.coms2 import Master_Session, assert_partial_wet
#from definitions import src_name


def rlay_extract(fp,
                 window=None, masked=False,
 
                 ):
    
    """load rlay data and arrays"""
    with rio.open(fp, mode='r') as ds:
        assert_rlay_simple(ds)
        stats_d = get_stats(ds) 
 
        ar = ds.read(1, window=window, masked=masked)
        
    return stats_d, ar 


class ValidateWorker(RioWrkr):
    """compute validation metrics for a raster by comparing to some true raster""" 
    
    confusion_ser=None
    confusion_codes = {'TP':111, 'TN':110, 'FP':101, 'FN':100}
    
    pred_fp=None
    true_fp=None
    
    def __init__(self,
                 true_fp=None,
                 pred_fp=None,
                 sample_pts_fp=None, 
                 logger=None,
                 **kwargs):
        
        #=======================================================================
        # pre init
        #=======================================================================
        if logger is None:
            logger=get_new_console_logger(level=logging.DEBUG)
        
        super().__init__(logger=logger,**kwargs)
                
        #=======================================================================
        # load rasters
        #=======================================================================
        """using ocnditional loading mostly for testing"""
        
        if not true_fp is None:
            self._load_true(true_fp) 
            
        if not pred_fp is None:            
            self._load_pred(pred_fp)
            
    def _load_true(self, true_fp):

        stats_d, self.true_ar = rlay_extract(true_fp)
        self.logger.info('loaded true raster from file w/\n    %s' % stats_d)
        #set the session defaults from this
        if 'dtypes' in stats_d:
            stats_d['dtype'] = stats_d['dtypes'][0]
        self.stats_d = stats_d
        self._set_defaults(stats_d)
        self.true_fp = true_fp
        
        if not self.pred_fp is None:
            assert_spatial_equal(self.true_fp, self.pred_fp)
            
            
        return


    def _load_pred(self, pred_fp):
        stats_d, self.pred_ar = rlay_extract(pred_fp)
        self.pred_fp=pred_fp
        
        if not self.true_fp is None:
            assert_spatial_equal(self.true_fp, self.pred_fp)
            
        self.logger.info('loaded pred raster from file w/\n    %s' % stats_d)
        


            
 
    
    #===========================================================================
    # grid inundation metrics------
    #===========================================================================
    def get_hitRate(self,**kwargs):
        """proportion of wet benchmark data that was replicated by the model"""
        #log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP']/(cf_ser['TP']+cf_ser['FN'])
    
    def get_falseAlarms(self,**kwargs):
        #log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['FP']/(cf_ser['TP']+cf_ser['FP'])
    
    def get_criticalSuccessIndex(self, **kwargs):
        """critical success index. accounts for both overprediction and underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP']/(cf_ser['TP']+cf_ser['FP']+cf_ser['FN'])
    
    def get_errorBias(self, **kwargs):
        """indicates whether the model has a tendency toward overprediction or underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        return cf_ser['FP']/cf_ser['FN']
    
    def get_inundation_all(self, **kwargs):
        """convenience for getting all the inundation metrics
        NOT using notation from Wing (2017)
        """
        
        d= {
            'hitRate':self.get_hitRate(**kwargs),
            'falseAlarms':self.get_falseAlarms(**kwargs),
            'criticalSuccessIndex':self.get_criticalSuccessIndex(**kwargs),
            'errorBias':self.get_errorBias(**kwargs),
            }
        
        self.logger.info('computed all inundation metrics:\n    %s'%d)
        return d
    
    def get_confusion_grid(self,
                           confusion_codes=None,**kwargs):
        """generate confusion grid
        
        Parameters
        ----------
        confusion_codes: dict
            integer codes for confusion labels
        """
        log, true_ar, pred_ar = self._func_setup_local('confuGrid', **kwargs)
        
        if confusion_codes is None: confusion_codes=self.confusion_codes
        
        #convert to boolean (true=wet=nonnull)
        true_arB, pred_arB = np.invert(np.isnan(true_ar)), np.invert(np.isnan(pred_ar))
        
        #start with dummy
        res_ar = np.full(true_ar.shape, np.nan)
        
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
        
        #=======================================================================
        # check
        #=======================================================================
        if __debug__:
            cf_ser = self._confusion(true_ar=true_ar, pred_ar=pred_ar, **kwargs)
            
            #build a frame with the codes
            df1 = pd.Series(res_ar.ravel(), name='grid_counts').value_counts().to_frame().reset_index()            
            df1['index'] = df1['index'].astype(int) 
            df2 = df1.join(pd.Series({v:k for k,v in confusion_codes.items()}, name='codes'), on='index'
                           ).set_index('index')
                           
            #join the values from sklearn calcs
            df3 = df2.join(cf_ser.rename('sklearn_counts').reset_index().rename(columns={'index':'codes'}).set_index('codes'), 
                     on='codes')
            
            compare_bx = df3['grid_counts']==df3['sklearn_counts']
            if not compare_bx.all():
                raise AssertionError('confusion count mismatch\n    %s'%compare_bx.to_dict())
            
        log.info('finished on %s'%str(res_ar.shape))
        
        return res_ar
    
    #===========================================================================
    # pipeline------
    #===========================================================================



    def run_vali(self,
                 true_fp=None, pred_fp=None,
                 write=True,
                 ):
        #=======================================================================
        # defaults
        #=======================================================================
        log=self.logger.getChild('vali')
        meta_lib = {'smry':{**{'today':self.today_str}, **self._get_init_pars()}}
        
        #=======================================================================
        # load
        #=======================================================================
        if not true_fp is None:
            self._load_true(true_fp) 
            
        if not pred_fp is None:            
            self._load_pred(pred_fp)
        
        assert isinstance(self.pred_ar, np.ndarray)
        #=======================================================================
        # grid metrics
        #=======================================================================
        shape, size = self.true_ar.shape, self.true_ar.size
        meta_lib['grid'] = {**{'shape':str(shape), 'size':size}, **copy.deepcopy(self.stats_d)}
        
        
        #=======================================================================
        # inundation metrics
        #=======================================================================
        log.info(f'computing inundation metrics on %s ({size})'%str(shape))
        confusion_ser = self._confusion()
        inun_metrics_d = self.get_inundation_all()
        
        #confusion grid
        confusion_grid_ar = self.get_confusion_grid()
        if write:
            confusion_grid_fp = self.write_array(confusion_grid_ar, resname = self._get_resname(dkey='confuGrid'))
        
        #meta
        meta_lib['inundation'] = {**confusion_ser.to_dict(), **inun_metrics_d, **{'confusion_grid_fp':confusion_grid_fp}}
        
        
        #=======================================================================
        # wrap
        #=======================================================================
        if write:
            self._write_meta(meta_lib, logger=log)
        
        return confusion_ser, inun_metrics_d
        
    #===========================================================================
    # private helpers------
    #===========================================================================
    def _confusion(self, **kwargs):
        """retrieve or construct the wet/dry confusion series"""
        if self.confusion_ser is None:
            log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
            
            #convert to boolean (true=wet=nonnull)
            true_arB, pred_arB = np.invert(np.isnan(true_ar)), np.invert(np.isnan(pred_ar))
            
            assert_partial_wet(true_arB)
            assert_partial_wet(pred_arB)
            
            """
            true_arB.sum()
            """
            
            #fancy labelling
            self.confusion_ser = pd.Series(confusion_matrix(true_arB.ravel(), pred_arB.ravel(),
                                                            labels=[False, True]).ravel(),
                      index = ['TN', 'FP', 'FN', 'TP'])
            
            log.info('generated wet-dry confusion matrix on %s\n    %s'%(
                str(true_ar.shape), self.confusion_ser.to_dict()))
            
        return self.confusion_ser.copy()

    def _func_setup_local(self, dkey, 
                    logger=None,  
                    true_ar=None, pred_ar=None,
                    ):
        """common function default setup
        
       
 
        
        """
 
        if logger is None:
            logger = self.logger
        log = logger.getChild(dkey)
        
        if true_ar is None: true_ar=self.true_ar
        if pred_ar is None: pred_ar=self.pred_ar
            
        return log, true_ar, pred_ar

    
class ValidateSession(ValidateWorker, RioSession, Master_Session):
    def __init__(self, 
                 run_name = None,
                 **kwargs):
 
        if run_name is None:
            run_name = 'vali_v1'
        super().__init__(run_name=run_name, **kwargs)
    



    
def run_validator(true_fp, pred_fp,        
        **kwargs):
    """compute error metrics and layers on a wse layer"""
    
    with ValidateSession(true_fp=true_fp, pred_fp=pred_fp, **kwargs) as ses:
        ses.run_vali()

        
    return ofp
        
        
    

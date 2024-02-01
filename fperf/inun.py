'''
Created on Jan. 6, 2023

@author: cefect

validating inundations
'''
import logging, os, copy, datetime, pickle
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
 
from sklearn.metrics import confusion_matrix

from hp.oop import Basic
from hp.rio import (
    RioSession, RioWrkr, assert_rlay_simple,  assert_spatial_equal, is_raster_file,
    write_array2,get_write_kwargs, rlay_ar_apply, get_meta, confusion_codes
    )

from hp.riom import load_mask_array, assert_mask_ar
 

from hp.gpd import (
    get_samples, GeoPandasWrkr,write_rasterize
    )
from hp.logr import get_new_console_logger
from hp.err_calc import ErrorCalcs, get_confusion_cat

from hp.hyd import (
    assert_partial_wet, assert_wse_ar, assert_wsh_ar, get_wsh_rlay
    )

from fperf.base import BaseWorker
 


# from definitions import src_name


class ValidateMask(BaseWorker, RioWrkr, Basic):
    """compute validation metrics for a inundation mask
    
    
    WARNING: use context management for multiple runs
    """ 
    
    confusion_ser = None
    
    
    pred_inun_fp = None
    true_inun_fp = None
    
 
    
    def __init__(self,
                 true_inun_fp=None,
                 pred_inun_fp=None,
 
                 logger=None,
                 **kwargs):
        """
        
        Pars
        ------------
 
            
        true_inun_fp: str
            filepath to inundation (rlay or poly) to compare extents against
            
        pred_inun_fp: str
            filepath to predicted/modelled inundation (rlay) to evaluate
        """
        
 
        #=======================================================================
        # pre init
        #=======================================================================
        if logger is None:
            logger = get_new_console_logger(level=logging.DEBUG)
        
        super().__init__(logger=logger, **kwargs)
                
        #=======================================================================
        # load rasters
        #=======================================================================
        """using ocnditional loading mostly for testing"""
        
        if not pred_inun_fp is None: 
            self._load_mask_pred(pred_inun_fp)
        
        if not true_inun_fp is None:
            if not is_raster_file(true_inun_fp):
                invert, maskType=True, 'native'
                
                #1=wet, -9999=dry
                rlay_fp = write_rasterize(true_inun_fp, pred_inun_fp)
            else:
                rlay_fp=true_inun_fp
                invert, maskType=True, 'binary'
                
            self._load_mask_true(rlay_fp, invert=invert, maskType=maskType) 
            
        self.confusion_codes=confusion_codes
            
        self.logger.debug('finished init')
            

            
    def _load_mask_true(self, rlay_fp, invert=True, maskType='binary'):
        """load the true mask
        
        true=inundated (because we load from a mask, need to invert)
        
        
        """
        assert is_raster_file(rlay_fp), 'must pass a raster file'
        
        
        #=======================================================================
        # if not maskType=='binary':
        #     """pipeline is setup for binaries... not sure what working with natives would do"""
        #     raise NotImplementedError(maskType)
        #=======================================================================
            
        stats_d, self.true_mar = get_rlay_mask(rlay_fp, invert=invert, maskType=maskType)
 
        self.logger.info('loaded true raster from file w/\n    %s' % stats_d)
        # set the session defaults from this
        if 'dtypes' in stats_d:
            stats_d['dtype'] = stats_d['dtypes'][0]
        self.stats_d = stats_d
        
        self._set_profile(rlay_fp)
        
        #self.true_inun_fp = rlay_fp
        
        if not self.pred_inun_fp is None:
            assert_spatial_equal(rlay_fp, self.pred_inun_fp)

    def _load_mask_pred(self, pred_fp, invert=True, maskType='binary'):
        """load the predicted mask
        
        Pars
        -------
        pred_fp: str
            filepath to mask-like raster
        """
        self.pred_stats_d, self.pred_mar = get_rlay_mask(pred_fp, invert=invert, maskType=maskType)
 
        
        #self.pred_inun_fp = pred_fp
        
        if not self.true_inun_fp is None:
            assert_spatial_equal(self.true_inun_fp, self.pred_inun_fp)
            
        self.logger.info('loaded pred raster from file w/\n    %s' % self.pred_stats_d)
        
 
 
    
    #===========================================================================
    # grid inundation metrics------
    #===========================================================================
    def get_hitRate(self, **kwargs):
        """proportion of wet benchmark data that was replicated by the model"""
        # log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP'] / (cf_ser['TP'] + cf_ser['FN'])
    
    def get_falseAlarms(self, **kwargs):
        # log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['FP'] / (cf_ser['TP'] + cf_ser['FP'])
    
    def get_criticalSuccessIndex(self, **kwargs):
        """critical success index. accounts for both overprediction and underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP'] / (cf_ser['TP'] + cf_ser['FP'] + cf_ser['FN'])
    
    def get_errorBias(self, **kwargs):
        """indicates whether the model has a tendency toward overprediction or underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        return cf_ser['FP'] / cf_ser['FN']
    
    def get_inundation_all(self, **kwargs):
        """convenience for getting all the inundation metrics
        NOT using notation from Wing (2017)
        """
        
        d = {
            'hitRate':self.get_hitRate(**kwargs),
            'falseAlarms':self.get_falseAlarms(**kwargs),
            'criticalSuccessIndex':self.get_criticalSuccessIndex(**kwargs),
            'errorBias':self.get_errorBias(**kwargs),
            }
        
        # add confusion codes
        d.update(self.confusion_ser.to_dict())
        
        #=======================================================================
        # #checks
        #=======================================================================
        assert set(d.keys()).symmetric_difference(
            ['hitRate', 'falseAlarms', 'criticalSuccessIndex', 'errorBias', 'TN', 'FP', 'FN', 'TP']
            ) == set()
            
        assert pd.Series(d).notna().all(), d
        
        self.logger.info('computed all inundation metrics:\n    %s' % d)
        return d

    def get_confusion_grid(self,
                           confusion_codes=None, **kwargs):
        """generate confusion grid
        
        Parameters
        ----------
        confusion_codes: dict
            integer codes for confusion labels
        """
        #(true=wet=nonnull)
        log, true_arB, pred_arB = self._func_setup_local('confuGrid', **kwargs)
        
        if confusion_codes is None: confusion_codes = self.confusion_codes
 
        #build a confusion map (using the int codes)
        res_ar = get_confusion_cat(true_arB, pred_arB, confusion_codes=confusion_codes)
        
        #=======================================================================
        # check
        #=======================================================================
        if __debug__:
            """compare against our aspatial confusion generator"""
            cf_ser = self._confusion(true_mar=true_arB, pred_mar=pred_arB, **kwargs)
            
            # build a frame with the codes
            df1 = pd.Series(res_ar.ravel(), name='grid_counts').value_counts().to_frame().reset_index()            
            df1['index'] = df1['index'].astype(int) 
            df2 = df1.join(pd.Series({v:k for k, v in confusion_codes.items()}, name='codes'), on='index'
                           ).set_index('index')
                           
            # join the values from sklearn calcs
            df3 = df2.join(cf_ser.rename('sklearn_counts').reset_index().rename(columns={'index':'codes'}).set_index('codes'),
                     on='codes')
            
            #compare
            if not df3['grid_counts'].sum()==df3['sklearn_counts'].sum():
                raise AssertionError('mismatch in confusion counts')
            
            compare_bx = df3['grid_counts'] == df3['sklearn_counts']
            if not compare_bx.all():
                raise AssertionError('confusion count mismatch\n    %s' % compare_bx.to_dict())
            
        log.info('finished on %s' % str(res_ar.shape))
        
        return res_ar
 
    #===========================================================================
    # private helpers------
    #===========================================================================
    def _confusion(self, **kwargs):
        """retrieve or construct the wet/dry confusion series"""
        if self.confusion_ser is None:
            log, true_mar, pred_mar = self._func_setup_local('hitRate', **kwargs)
            
 
            # fancy labelling
            cser = pd.Series(confusion_matrix(true_mar.ravel(), pred_mar.ravel(),
                                                            labels=[False, True]).ravel(),
                      index=['TN', 'FP', 'FN', 'TP'])
            
            assert cser.notna().all(), cser
            
            log.info('generated wet-dry confusion matrix on %s\n    %s' % (
                str(true_mar.shape), cser.to_dict()))
            
            self.confusion_ser = cser.copy()
            
        return self.confusion_ser.copy()

    def _func_setup_local(self, dkey,
                    logger=None,
                    true_mar=None, pred_mar=None,
                    ):
        """common function default setup"""
 
        if logger is None:
            logger = self.logger
        log = logger.getChild(dkey)
        
        if true_mar is None: true_mar = self.true_mar
        if pred_mar is None: pred_mar = self.pred_mar
        
        #(true=wet=nonnull)
        assert_partial_wet(true_mar, msg='true mask')
        assert_partial_wet(pred_mar, msg='pred mask')
            
        return log, true_mar, pred_mar
    
    def _check_inun(self, **kwargs):
        """check loaded inundation grids"""
        
        log, true_mar, pred_mar = self._func_setup_local('check', **kwargs)
        
        # data difference check
        if (true_mar == pred_mar).all():
            raise AssertionError('passed identical grids')
 
        
    



def get_rlay_mask(fp,
                   invert=True,
                 **kwargs):
    """load the mask from a raster and some stats"""
    
    #===========================================================================
    # retrieve
    #===========================================================================
    bool_ar = load_mask_array(fp, **kwargs)
    assert_mask_ar(bool_ar)
    stats_d = get_meta(fp)
    stats_d['null_cnt'] = bool_ar.sum()
        
    #checks
    if invert:
        mask = np.invert(bool_ar)
    else:
        mask = bool_ar
        
    assert_partial_wet(mask, msg=os.path.basename(fp))
    
    return stats_d, mask
    

    

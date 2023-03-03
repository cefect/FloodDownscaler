'''
Created on Mar. 3, 2023

@author: cefect
'''
import logging, os, copy, datetime, pickle
import numpy as np
from hp.rio import (
    RioSession, RioWrkr, assert_rlay_simple, get_stats, assert_spatial_equal, get_depth,is_raster_file,
    write_array2,get_write_kwargs, rlay_ar_apply
    )

from fdsc.base import (
    Master_Session, assert_partial_wet, rlay_extract, assert_wse_ar, assert_wd_ar
    )

from fdsc.analysis.valid.v_inun import ValidateMask
from fdsc.analysis.valid.v_wd import ValidatePoints

class ValidateSession(ValidateMask, ValidatePoints, RioSession, Master_Session):

    def __init__(self,
                 run_name=None,
                 **kwargs):
 
        if run_name is None:
            run_name = 'vali_v1'
        super().__init__(run_name=run_name, **kwargs)

    def run_vali_pts(self, sample_pts_fp,
                            true_fp=None,
                           pred_fp=None,
                           **kwargs):
        """validation on poitns pipeline
        assumes rasters are depth rasters"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('pts', subdir=False, ext='.gpkg', **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir)
 
        #=======================================================================
        # #sample points
        #=======================================================================
        gdf = self.get_samples(true_fp=true_fp, pred_fp=pred_fp, sample_pts_fp=sample_pts_fp, **skwargs)
        
        # write
        meta_d = {'sample_pts_fp':sample_pts_fp, 'cnt':len(gdf)}
        # ofpi = self._get_ofp(out_dir=out_dir, dkey='samples', ext='.geojson')
        gdf.to_file(ofp, crs=self.crs)
        meta_d['samples_fp'] = ofp
        log.info(f'wrote {len(gdf)} to \n    {ofp}')
        #=======================================================================
        # #calc errors
        #=======================================================================
        err_d = self.get_samp_errs(gdf, **skwargs)
        # meta
        meta_d.update(err_d)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        return err_d, meta_d

    def run_vali(self,
                 wse_true_fp=None, 
                 inun_true_fp=None,
                 pred_fp=None,
                 sample_pts_fp=None, dem_fp=None,
                 write_meta=True,
                 **kwargs):
        """
        run all validations on a downsampled grid (compared to a true grid).
            called by pipeline.run_dsc_vali()
        
        
        Parameters
        -----------

        
        sample_pts_fp: str, optional
            filepath to points vector layer for sample-based metrics
            
        dem_fp: str, optional
            filepath to dem (for converting to depths)
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('vali', subdir=True, **kwargs)
        meta_lib = {'smry':{**{'today':self.today_str}, **self._get_init_pars()}}
        metric_lib = dict()
        skwargs = dict(logger=log)
        #=======================================================================
        # load
        #=======================================================================
        if not wse_true_fp is None:
            self._load_mask_true(wse_true_fp) 
        else:
            wse_true_fp = self.true_inun_fp
            
        if not pred_fp is None: 
            self._load_mask_pred(pred_fp)
        else:
            pred_fp = self.pred_fp
            
        if sample_pts_fp is None:
            sample_pts_fp = self.sample_pts_fp
            
        if dem_fp is None:
            dem_fp = self.dem_fp
        
        #=======================================================================
        # precheck
        #=======================================================================
        true_ar, pred_ar = self.true_ar, self.pred_mar
        assert isinstance(pred_ar, np.ndarray)
        
        # data difference check
        if (true_ar == pred_ar).all():
            raise IOError('passed identical grids')
        if (true_ar.mask == pred_ar.mask).all():
            log.warning('identical masks on pred and true')
        
        #=======================================================================
        # grid metrics
        #=======================================================================        
        shape, size = true_ar.shape, true_ar.size
        meta_lib['grid'] = {**{'shape':str(shape), 'size':size, 'wse_true_fp':wse_true_fp, 'pred_fp':pred_fp},
                            **copy.deepcopy(self.stats_d)}
        
        #=======================================================================
        # inundation metrics-------
        #=======================================================================
        log.info(f'computing inundation metrics on %s ({size})' % str(shape))
        
        # confusion_ser = self._confusion(**skwargs)
        inun_metrics_d = self.get_inundation_all(**skwargs)
        
        # confusion grid
        confusion_grid_ar = self.get_confusion_grid(**skwargs)
 
        # meta
        meta_d = copy.deepcopy(inun_metrics_d)
        
        #=======================================================================
        # write
        #=======================================================================
        meta_d['confuGrid_fp'] = self.write_array(confusion_grid_ar, out_dir=out_dir,
                                                       resname=self._get_resname(dkey='confuGrid'))
 
        meta_lib['inun'] = meta_d
        metric_lib['inun'] = inun_metrics_d
        
        #=======================================================================
        # asset samples---------
        #=======================================================================            
        if not sample_pts_fp is None:
            assert isinstance(dem_fp, str), type(dem_fp)
            # build depth grids
            true_dep_fp = get_depth(dem_fp, wse_true_fp, ofp=self._get_ofp(out_dir=out_dir, resname='true_dep'))
            pred_dep_fp = get_depth(dem_fp, pred_fp, ofp=self._get_ofp(out_dir=tmp_dir, resname='pred_dep'))
            
            metric_lib['samp'], meta_lib['samp'] = self.run_vali_pts(sample_pts_fp,
                                        wse_true_fp=true_dep_fp, pred_fp=pred_dep_fp, logger=log, out_dir=out_dir)
            
            meta_lib['grid']['true_dep_fp'] = true_dep_fp
            meta_lib['grid']['dep1'] = pred_dep_fp
        
        #=======================================================================
        # wrap-----
        #=======================================================================        
        if write_meta:
            self._write_meta(meta_lib, logger=log, out_dir=out_dir)
        
        log.info('finished')
        return metric_lib, meta_lib
    
    

def run_validator(true_fp, pred_fp,
        **kwargs):
    """compute error metrics and layers on a wse layer"""
    
    with ValidateSession(true_fp=true_fp, pred_fp=pred_fp, **kwargs) as ses:
        res = ses.run_vali()
        
    return res

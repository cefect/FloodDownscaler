'''
Created on Jan. 25, 2023

@author: cefect

simple downscaling tools
'''

import os, datetime, shutil
import numpy as np
import numpy.ma as ma
import pandas as pd
 
import rasterio as rio
from rasterio import shutil as rshutil


from hp.basic import now
from hp.gdal import getNoDataCount
from hp.rio import (
    assert_extent_equal, assert_ds_attribute_match, _get_meta, assert_rlay_simple, RioSession,
    write_array, assert_spatial_equal, get_write_kwargs, rlay_calc1, load_array, write_clip,
    rlay_apply,rlay_ar_apply,write_resample, Resampling, get_ds_attr, get_stats2
    )
from hp.riom import write_extract_mask, write_array_mask
 

from fdsc.base import (
    assert_dem_ar, assert_wse_ar, rlay_extract, assert_partial_wet, assert_type_fp, DscBaseWorker
    )





class WetPartials(DscBaseWorker):
    """first phase of two phase downsamplers"""


    def p1_wetPartials(self, wse2_fp, dem_fp, downscale=None,
                       resampling=Resampling.bilinear,
                       dem_filter=True,
                        **kwargs):
        """downscale wse2 grid in wet-partial regions
        
        Parameters
        ------------
        dem_filter: bool, default True
            whether or not to filter by DEM
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('p1WP', subdir=True, **kwargs)
        start = now()
        if downscale is None: 
            downscale = self.get_downscale(wse2_fp, dem_fp)

        log.info(f'downscale={downscale} w/ dem_filter={dem_filter} on {os.path.basename(wse2_fp)} w/ {resampling}')
        #=======================================================================
        # #precheck
        #=======================================================================
        assert_extent_equal(wse2_fp, dem_fp, msg='phase1')
 
        meta_d = {'wse2_fp':wse2_fp, 'dem_fp':dem_fp, 'resampling':resampling, 'downscale':downscale}
 
        #=======================================================================
        # resample
        #=======================================================================
 
        wse1_rsmp_fp = write_resample(wse2_fp, resampling=resampling,
                       scale=downscale,
                       ofp=self._get_ofp(dkey='resamp', out_dir=tmp_dir, ext='.tif'),
                       )
        
        assert_type_fp(wse1_rsmp_fp, 'WSE', msg=f'resample {downscale}')
        #rlay_ar_apply(wse1_rsmp_fp, assert_wse_ar, msg='WSE resample')
        
        meta_d['wse1_rsmp_fp'] = wse1_rsmp_fp
 
        #=======================================================================
        # #filter dem violators
        #=======================================================================
        if dem_filter:
            wse1_filter_ofp, d = self.get_wse_dem_filter(wse1_rsmp_fp, dem_fp, logger=log, out_dir=tmp_dir)
            meta_d.update(d)
        else:
            wse1_filter_ofp = wse1_rsmp_fp
 
        #=======================================================================
        # wrap
        #=======================================================================
        rshutil.copy(wse1_filter_ofp, ofp)
        
        tdelta = (now() - start).total_seconds()
        meta_d['tdelta'] = tdelta
        
        log.info(f'built wse from downscale={downscale} on wet partials\n    {meta_d}\n    {ofp}')
        meta_d['wse1_wp_fp'] = ofp
        return ofp, meta_d

class BasicDSC(WetPartials):
    
    def __init__(self,
                 run_dsc_handle_d=None, 
                 **kwargs):
        
        if run_dsc_handle_d is None: run_dsc_handle_d=dict()
        
        run_dsc_handle_d['Basic'] = self.run_rsmp #add your main method to the caller dict
        run_dsc_handle_d['SimpleFilter'] = self.run_rsmpF
        
        super().__init__(run_dsc_handle_d=run_dsc_handle_d, **kwargs)
        
    def run_rsmp(self,**kwargs):
        """run Basic (resample only) 
        """
        return self.run_p1(method='Basic', dem_filter=False, **kwargs)
    
    def run_rsmpF(self,**kwargs):
        """run Basic (resample only) + DEM filter     
        """
        return self.run_p1(method='SimpleFilter', dem_filter=True, **kwargs)
        
    def run_p1(self,wse_fp=None, dem_fp=None, method='Basic', dem_filter=True, **kwargs):
        """just a wrapper for p1_wetPartials"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup(self.nicknames_d[method], 
                                                               subdir=False, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir, ofp=ofp)
        meta_lib={'smry':dict()}
        downscale = self.downscale
        
        #call pse 1
        wse1_wp_fp, meta_lib['p1_wp'] = self.p1_wetPartials(wse_fp, 
                dem_fp, downscale=downscale,dem_filter=dem_filter, **skwargs)
        
        
 
        return wse1_wp_fp, meta_lib
        
        
 


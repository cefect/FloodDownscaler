'''
Created on Mar. 3, 2023

@author: cefect

validating water depths
'''
#===============================================================================
# IMPORTS-------
#===============================================================================
import logging, os, copy, datetime, pickle
import numpy as np
import geopandas as gpd

import pandas as pd
import rasterio as rio

from hp.logr import get_new_console_logger
from hp.rio import (
    RioSession, RioWrkr, assert_rlay_simple, get_stats, assert_spatial_equal, get_depth,is_raster_file,
    write_array2,get_write_kwargs, rlay_ar_apply
    )

from hp.gpd import (
    get_samples, GeoPandasWrkr,write_rasterize
    )

from hp.err_calc import ErrorCalcs, get_confusion_cat

from fdsc.base import (
    Master_Session, assert_partial_wet, rlay_extract, assert_wse_ar, assert_wd_ar
    )

class ValidatePoints(RioWrkr, GeoPandasWrkr):
    """methods for validation with points"""
    
    stats_d = None
    pts_gdf = None
    sample_pts_fp = None
    dem_fp = None
    
    def __init__(self,
                 true_wd_fp=None,
                 pred_wd_fp=None,
                 
                 
 
                 sample_pts_fp=None,
 
                 index_coln='id',wd_key='water_depth',
                 logger=None,

                 **kwargs):
        """
        Pars
        ---------
        true_wd_fp: str
            filepath to raster of WD values to validate against
            
        """
        
        #=======================================================================
        # pre init
        #=======================================================================
        if logger is None:
            logger = get_new_console_logger(level=logging.DEBUG)
        
        super().__init__(logger=logger, **kwargs)
        
        #=======================================================================
        # attach
        #=======================================================================
        # depth rasters
        if not true_wd_fp is None:
            rlay_ar_apply(true_wd_fp, assert_wd_ar)
            
            self.true_wd_fp=true_wd_fp
            
        if not pred_wd_fp is None:
            rlay_ar_apply(pred_wd_fp, assert_wd_ar)
            
            self.pred_wd_fp=pred_wd_fp
            
        
        self.wd_key=wd_key
        self.index_coln = index_coln
        #=======================================================================
        # load 
        #=======================================================================
 
        if not sample_pts_fp is None:
            self._load_pts(sample_pts_fp, index_coln=index_coln)
            
        
            
    def _load_stats(self, fp=None):
        """set session stats from a raster
        
        mostly used by tests where we dont load the raster during init"""
            
        assert not fp is None
        
        with rio.open(fp, mode='r') as ds:
            rlay_ar_apply(ds, assert_wd_ar)
            assert_rlay_simple(ds)
            self.stats_d = get_stats(ds) 
 

    def _get_gdf(self, fp, index_coln=None, bbox=None, stats_d=None):
        
        #=======================================================================
        # defaults
        #=======================================================================
        assert os.path.exists(fp)
        
        if index_coln is None: index_coln = self.index_coln
        
        if stats_d is None:
            if self.stats_d is None: self._load_stats()
            stats_d = self.stats_d.copy()
        
    # get bounding box from rasters
        if bbox is None:
            bbox = stats_d['bounds']
            
        #=======================================================================
        # load
        #=======================================================================
        gdf = gpd.read_file(fp, bbox=bbox)
        # check
        assert gdf.crs == stats_d['crs'], f'crs mismatch between points {gdf.crs} and raster %s' % stats_d['crs']
        assert (gdf.geometry.geom_type == 'Point').all()
        # clean
        return gdf.set_index(index_coln)

    def _load_pts(self, fp, **kwargs):
        """load sample points"""
 
        
        self.pts_gser = self._get_gdf(fp, **kwargs).geometry
        self.sample_pts_fp = fp
        
    def get_samples(self,
                           true_wd_fp=None,
                           pred_wd_fp=None,
                           sample_pts_fp=None,
                           gser=None,
                           **kwargs):
        """sample raster with poitns"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('samps', **kwargs)
        
        if true_wd_fp is None:
            true_wd_fp = self.true_wd_fp
        if pred_wd_fp is None:
            pred_wd_fp = self.pred_wd_fp
            
        if not sample_pts_fp is None:
            assert gser is None
            self._load_stats(true_wd_fp)
            self._load_pts(sample_pts_fp)
            
        if gser is None:
            gser = self.pts_gser
 
        #=======================================================================
        # sample each
        #=======================================================================
        log.info(f'sampling {len(gser)} pts on 2 rasters')
        d = dict()
        for k, fp in {'true':true_wd_fp, 'pred':pred_wd_fp}.items():
            log.info(f'sampling {k}')
            with rio.open(fp, mode='r') as ds:
                d[k] = get_samples(gser, ds, colName=k).drop('geometry', axis=1)
        
        samp_gdf = pd.concat(d.values(), axis=1).set_geometry(gser)
        
        #=======================================================================
        # wrap
        #=======================================================================
        assert not samp_gdf.isna().any().any(), 'no nulls.. should be depths'
        
        log.info(f'finished sampling w/ {str(samp_gdf.shape)}')
        
        return samp_gdf
    
    def get_samp_errs(self, gdf_raw, **kwargs):
        """calc errors between pred and true"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('samp_errs', **kwargs)
        
        #=======================================================================
        # clean
        #=======================================================================
        gdf = gdf_raw.drop('geometry', axis=1)  # .dropna(how='any', subset=['true'])
        
        assert gdf.notna().all().all()
        
        #=======================================================================
        # calc
        #=======================================================================
        
        with ErrorCalcs(pred_ser=gdf['pred'], true_ser=gdf['true'], logger=log) as wrkr:
            err_d = wrkr.get_all(dkeys_l=['bias', 'meanError', 'meanErrorAbs', 'RMSE', 'pearson'])
            
            # get confusion
            _, cm_dx = wrkr.get_confusion(wetdry=True, normed=False)            
            err_d.update(cm_dx.droplevel([1, 2])['counts'].to_dict())
            
        return err_d

    

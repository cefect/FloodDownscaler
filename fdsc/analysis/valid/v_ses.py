'''
Created on Mar. 3, 2023

@author: cefect
'''
import logging, os, copy, datetime, pickle
import numpy as np
import shapely.geometry as sgeo
import geopandas as gpd
import rasterio as rio
from hp.rio import (
    RioSession, RioWrkr, assert_rlay_simple, get_stats, assert_spatial_equal, get_depth,is_raster_file,
    write_array2,get_write_kwargs, rlay_ar_apply, get_meta, get_ds_attr, write_clip
    )

from hp.gpd import (
    write_rasterize, get_samples
    )

from fdsc.base import (
    Master_Session, assert_partial_wet, rlay_extract, assert_wse_ar, assert_wd_ar, assert_dem_ar
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
        
        
    def run_vali_inun(self,
               true_inun_fp=None,
                 pred_inun_fp=None,
                 
                 **kwargs):
        
        """run inundation validation sequence"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('inun', subdir=True, **kwargs)
 
        skwargs = dict(logger=log) #these funcs are using a specical local setup
 
        #=======================================================================
        # load
        #=======================================================================
        if not true_inun_fp is None:
            self._load_mask_true(true_inun_fp) 
            
        if not pred_inun_fp is None: 
            self._load_mask_pred(pred_inun_fp)
 
        mar = self.true_mar
        #=======================================================================
        # precheck
        #=======================================================================
        self._check_inun()
        
 
        
        #=======================================================================
        # inundation metrics-------
        #=======================================================================
        log.info(f'computing inundation metrics on {mar.shape} ({mar.size})')
        
        # confusion_ser = self._confusion(**skwargs)
        inun_metrics = self.get_inundation_all(**skwargs)
        
        # confusion grid
        confusion_grid_ar = self.get_confusion_grid(**skwargs)
        
        confuGrid_fp = self.write_array(confusion_grid_ar, out_dir=out_dir,resname=self._get_resname(dkey='confuGrid'))
        
 
            
            
        #=======================================================================
        # wrap
        #=======================================================================
        
        log.info(f'finished w/ \n    {inun_metrics}')
        
        return inun_metrics, confuGrid_fp
    
    def run_vali_confuSamps(self,
                            confuGrid_fp, sample_pts_fp,
                            **kwargs):
        """sample confusion grid with points"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('confuS', subdir=True, ext='.gpkg', **kwargs)
        
        log.info(f'computing inundation metrics on samples from {os.path.basename(sample_pts_fp)}')
        
        #load points
        gdf = self._get_gdf(sample_pts_fp, stats_d=get_meta(confuGrid_fp))
        
        
        # get values from raster 
        log.info(f'sampling {os.path.basename(confuGrid_fp)} on points')
        with rio.open(confuGrid_fp, mode='r') as ds:
            confu_gdf = get_samples(gdf.geometry, ds, colName='confusion')
            
        #=======================================================================
        # #write
        #=======================================================================
 
        confu_gdf.to_file(ofp, crs=self.crs)
 
        log.info(f'wrote {len(gdf)} to \n    {ofp}')
        
        return ofp
        
 
 

    def run_vali_pts(self, sample_pts_fp,
                           true_wd_fp=None,
                           pred_wd_fp=None,
                           **kwargs):
        """comprae two depth rasters at points"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('pts', subdir=True, ext='.gpkg', **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir)
 
        #=======================================================================
        # #sample points
        #=======================================================================
        gdf = self.get_samples(true_wd_fp=true_wd_fp, pred_wd_fp=pred_wd_fp, sample_pts_fp=sample_pts_fp, **skwargs)
        
        # write
        meta_d = {'sample_pts_fp':sample_pts_fp, 'cnt':len(gdf)}
 
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
    
    def run_vali_hwm(self, wd_fp, hwm_fp, wd_key=None,
                     **kwargs):
        """compare a depth raster against some point values"""
        #=======================================================================
        # defautls
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('hwm', subdir=True, ext='.gpkg', **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir)
        if wd_key is None: wd_key=self.wd_key
        
        #=======================================================================
        # load points
        #=======================================================================
        rlay_stats_d = get_meta(wd_fp)
        
        gdf = self._get_gdf(hwm_fp, stats_d=rlay_stats_d)
        
        assert wd_key in gdf
        
        log.info(f'loaded {len(gdf)} HWMs from \n    {hwm_fp}') 
        
        #=======================================================================
        # get values from raster
        #=======================================================================
        log.info(f'sampling {os.path.basename(wd_fp)} on points')
        with rio.open(wd_fp, mode='r') as ds:
            gdf = gdf.join(
                get_samples(gdf.geometry, ds, colName='pred').drop('geometry', axis=1)
                ).drop('geometry', axis=1).set_geometry(gdf.geometry).rename(columns={wd_key:'true'})
                
        
        # write
        meta_d = {'hwm_pts_fp':hwm_fp, 'cnt':len(gdf), 'wd_fp':wd_fp}
 
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
                 pred_wse_fp=None, 
                 true_wse_fp=None,
                 true_inun_fp=None,
                 sample_pts_fp=None,
                 hwm_pts_fp=None,
                 dem_fp=None,
                 write_meta=True,
                 **kwargs):
        """
        run all validations on a downsampled grid (compared to a true grid).
            called by pipeline.run_dsc_vali()
            
            allows separate inundation and wse validation
                or just uses wse
                
            allows inundation to be a polygon
        
        
        Parameters
        -----------
        pred_wse_fp: str
            predicted WSE grid. used to build WD. 
            
        true_wse_fp: str
            valid WSE grid. used to build inun grid if its not passed
            
        true_inun_fp: str, optional
            valid inundation extents (rlay or vlay). uses true_wse_fp if not passed
        
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
        skwargs = dict(logger=log, out_dir=out_dir)
        fp_d = dict(dem_fp=dem_fp, pred_wse_fp=pred_wse_fp, true_wse_fp=true_wse_fp, 
                    true_inun_fp=true_inun_fp,  hwm_pts_fp=hwm_pts_fp) #for reporting
        #=======================================================================
        # common prep
        #=======================================================================
        #dem
        assert isinstance(dem_fp, str), type(dem_fp)
        rlay_ar_apply(dem_fp, assert_dem_ar)        
 
        #pred
        if pred_wse_fp is None:
            raise NotImplementedError('need to passe a wse')
    
        meta_lib['grid'] = get_meta(pred_wse_fp)
        
        rlay_ar_apply(pred_wse_fp, assert_wse_ar)
        
        assert_spatial_equal(dem_fp, pred_wse_fp)
        
        #helper func
        def clip_rlay(rlay_fp):
            """clip the raster to the predicted bounds if necessary"""
            pd = meta_lib['grid']
            ibounds = get_ds_attr(rlay_fp, 'bounds')
            
            if ibounds == pd['bounds']:
                return rlay_fp
            else:
                ofp, stats_d = write_clip(rlay_fp,
                                  ofp=os.path.join(tmp_dir, os.path.basename(rlay_fp).replace('.tif', '_clip.tif')), 
                                  bbox=sgeo.box(*pd['bounds']), crs=pd['crs'])
                
                return ofp
            
        #=======================================================================
        # get depths
        #=======================================================================
        """doing this everytime... nice to have the wd for plots"""
        #if (sample_pts_fp!=None) or (hwm_pts_fp!=None):
        #predicted
        pred_wd_fp = get_depth(dem_fp, pred_wse_fp, out_dir=out_dir)
        rlay_ar_apply(pred_wd_fp, assert_wd_ar, msg='pred')
        
        self.pred_wd_fp=pred_wd_fp
        
        fp_d['pred_wd_fp'] = pred_wd_fp
            
        
        #=======================================================================
        # WD samples between grids----
        #======================================================================= 
        if (not sample_pts_fp is None) and (not true_wse_fp is None):          
            log.info(f'computing WD performance at points: \n    {sample_pts_fp}')
 
            #===================================================================
            # #build depths arrays
            #===================================================================                        
            true_wd_fp = get_depth(dem_fp, clip_rlay(true_wse_fp), out_dir=tmp_dir)
            rlay_ar_apply(true_wd_fp, assert_wd_ar, msg='true')
            self.true_wd_fp=true_wd_fp
            fp_d['true_wd_fp'] = true_wd_fp
            #===================================================================
            # run
            #===================================================================
            metric_lib['pts'], meta_lib['pts'] = self.run_vali_pts(sample_pts_fp,
                                        true_wd_fp=true_wd_fp, pred_wd_fp=pred_wd_fp, 
                                        **skwargs)
            
            fp_d['pts_samples_fp'] = meta_lib['pts']['samples_fp']
            
        #=======================================================================
        # HWMs--------
        #=======================================================================
        if not hwm_pts_fp is None:
            log.info(f'computing performance against HWMs ({os.path.basename(hwm_pts_fp)})')
            
            metric_lib['hwm'], meta_lib['hwm'] = self.run_vali_hwm(pred_wd_fp, hwm_pts_fp, **skwargs)
            
            fp_d['hwm_samples_fp'] = meta_lib['hwm']['samples_fp']
 
        
        #=======================================================================
        # inundatdion extents--------
        #=======================================================================
 
        #=======================================================================
        # true inundation
        #=======================================================================
        if true_inun_fp is None:
            log.info('using \'true_wse_fp\' for inundation validation')
            true_inun_fp = clip_rlay(true_wse_fp)
            
        # rasterize
        if not is_raster_file(true_inun_fp):
            log.info('rasterizing polygon')
            true_inun_rlay_fp = write_rasterize(true_inun_fp, pred_wse_fp)
            
        else:
            true_inun_rlay_fp = true_inun_fp
        
        fp_d['true_inun_rlay_fp'] = true_inun_rlay_fp
 
        #=======================================================================
        # run
        #=======================================================================
        metric_lib['inun'], confuGrid_fp = self.run_vali_inun(true_inun_fp=true_inun_rlay_fp, pred_inun_fp=pred_wse_fp, **skwargs)        
        meta_lib['inun_metrics'] = metric_lib['inun']
        fp_d['confuGrid_fp'] = confuGrid_fp
        
        #=======================================================================
        # sample with points
        #=======================================================================
        if not sample_pts_fp is None:
            fp_d['confuSamps_fp'] = self.run_vali_confuSamps(confuGrid_fp, sample_pts_fp, **skwargs)
        #=======================================================================
        # wrap-----
        #=======================================================================
        meta_lib['fps'] = fp_d
        if write_meta:
            self._write_meta(meta_lib, logger=log, out_dir=out_dir)
        
        log.info('finished')
        return metric_lib, meta_lib
    
    

def run_validator(rkwargs,**kwargs):
    """compute error metrics and layers on a wse layer"""
    
    with ValidateSession(**kwargs) as ses:
        return ses.run_vali(**rkwargs)
 

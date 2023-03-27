'''
Created on Dec. 4, 2022

@author: cefect

flood downscaling top-level control scripts
'''
import os, datetime, shutil, pickle
import numpy as np
import numpy.ma as ma
 
import rasterio as rio
from rasterio import shutil as rshutil

from hp.rio import (
    assert_extent_equal, assert_ds_attribute_match, get_stats, assert_rlay_simple, RioSession,
    write_array, assert_spatial_equal, get_write_kwargs, rlay_calc1, load_array, write_clip,
    rlay_apply, rlay_ar_apply, write_resample, Resampling, get_ds_attr, get_stats2
    )
from hp.pd import view, pd
from hp.gdal import getNoDataCount
from hp.hyd import (
    assert_type_fp
    )

from fdsc.wbt import WBT_worker
from fdsc.base import (
    Master_Session, assert_dem_ar, assert_wse_ar, rlay_extract, now
    )

from fdsc.simple import BasicDSC
from fdsc.schu14 import Schuman14
from fdsc.costGrow import CostGrow
from fdsc.bufferLoop import BufferGrowLoop





class Dsc_Session(CostGrow, BufferGrowLoop, Schuman14,BasicDSC,
        RioSession, Master_Session, WBT_worker):
      
    #===========================================================================
    # phase0-------  
    #===========================================================================
    def p0_clip_rasters(self, wse_fp, dem_fp,
                        bbox=None, crs=None,
                        **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('clip_rasters', **kwargs) 
     
        write_kwargs = RioSession._get_defaults(self, bbox=bbox, crs=crs, as_dict=True)
        bbox = write_kwargs['bbox']
 
        #=======================================================================
        # clip wse
        #=======================================================================
        wse_clip_fp, wse_stats = write_clip(wse_fp, ofp=os.path.join(tmp_dir, 'wse2_clip.tif'), **write_kwargs)
        
        dem_clip_fp, dem_stats = write_clip(dem_fp, ofp=os.path.join(tmp_dir, 'dem1_clip.tif'), **write_kwargs)
 
        #=======================================================================
        # warp
        #=======================================================================
 
        log.info(f'clipped rasters and wrote to\n    {tmp_dir}\n    {bbox.bounds}')
        return wse_clip_fp, dem_clip_fp
        
    def p0_load_rasters(self, wse2_rlay_fp, dem1_rlay_fp, crs=None,
                          **kwargs):
        """load and extract some data from the raster files"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('load_rasters', **kwargs)
        crs, bbox, compress, nodata = RioSession._get_defaults(self, crs=crs)

        #=======================================================================
        # load
        #=======================================================================
        # load wse with aoi rounded
        wse_stats, wse2_ar = rlay_extract(wse2_rlay_fp)
 
        dem_stats, dem1_ar = rlay_extract(dem1_rlay_fp)        
        s2, s1 = dem_stats['res'][0], wse_stats['res'][0]
        
        #=======================================================================
        # check
        #=======================================================================
        assert_dem_ar(dem1_ar)
        assert_wse_ar(wse2_ar)
        
        if crs is None:
            self.crs = dem_stats['crs']
            crs = self.crs
            
            log.info('set crs from dem (%s)' % crs.to_epsg())
            
        assert dem_stats['crs'] == crs, f'DEM crs %s doesnt match session {crs}' % dem_stats['crs']
        for stat in ['crs', 'bounds']:
            assert dem_stats[stat] == wse_stats[stat]
 
        assert s2 < s1, 'dem must have a finer resolution than the wse'
        if not s1 % s2 == 0.0:
            log.warning(f'uneven resolution relation ({s1}/{s2}={s1%s2})')
            
        # report
        downscale = s1 / s2
        log.info(f'downscaling from {s2} to {s1} ({downscale})')
 
        #=======================================================================
        # wrap
        #=======================================================================
     
      
        self.s2, self.s1, self.downscale = s2, s1, downscale 
        return wse2_ar, dem1_ar, wse_stats, dem_stats
    
 

    def run_dsc(self,
            dem1_fp,
            wse2_fp,
            
 
            method='CostGrow',
            downscale=None,
            write_meta=True,
            subdir=True,
            rkwargs=dict(),
            debug=None,
                **kwargs):
        """run a downsampling pipeline
        
        Paramerters
        -------------
        wse2_fp: str
            filepath to WSE raster layer at low-resolution (to be downscaled)
            
        dem1_fp: str
            filepath to DEM raster layer at high-resolution (used to infer downscaled WSE)
            
        method: str
            downsccaling method to apply
            
        debug: bool, None
            optional flag to disable debugging on just this method (nice for run_dsc_multi)
            
        Note
        -------
        no AOI clipping is performed. raster layers must have the same spatial extents. 
        see p0_clip_rasters to pre-clip the rasters
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dsc', subdir=subdir, **kwargs)
        if debug is None: debug=__debug__
        
        meta_lib = {'smry':{**{'today':self.today_str, 'method':method, 'wse2_fp':os.path.basename(wse2_fp), 'dem_fp':dem1_fp, 'ofp':ofp}, 
                            **self._get_init_pars()}}
        
        skwargs = dict(logger=log, out_dir=out_dir, tmp_dir=tmp_dir)
        start = now()
        
        
        
        #=======================================================================
        # precheck and load rasters
        #=======================================================================
        assert not os.path.exists(ofp)
        if debug:
            assert_extent_equal(wse2_fp, dem1_fp)
            assert_type_fp(dem1_fp, 'DEM')
            assert_type_fp(wse2_fp, 'WSE')
 
        meta_lib['wse_raw'] = get_stats2(wse2_fp)
        meta_lib['dem_raw'] = get_stats2(dem1_fp)
        
        if downscale is None:
            downscale = self.get_downscale(wse2_fp, dem1_fp)
        
        self.downscale=downscale
        meta_lib['smry']['downscale']=downscale
        #=======================================================================
        # run algo
        #=======================================================================
        f = self.run_dsc_handle_d[method]
        
        wse1_fp, d = f(wse_fp=wse2_fp, dem_fp=dem1_fp, **rkwargs, **skwargs)
        #=======================================================================
        # try:
        #     wse1_fp, meta_lib = f(wse_fp=wse2_fp, dem_fp=dem1_fp, **skwargs)
        # except Exception as e:
        #     raise IOError(f'failed to execute algo \'{method}\' method \'{f.__name__}\' w/ \n    {e}')
        #=======================================================================
        meta_lib.update(d)
 
        #=======================================================================
        # check
        #=======================================================================
        assert_type_fp(wse1_fp, 'WSE')
        assert_spatial_equal(wse1_fp, dem1_fp)            
            
        #=======================================================================
        # wrap
        #=======================================================================
        # copy tover to the main result
        if not ofp==wse1_fp:
            rshutil.copy(wse1_fp, ofp)
        
        tdelta = (now() - start).total_seconds()
        meta_lib['smry']['tdelta'] = tdelta
 
        if write_meta:
            self._write_meta(meta_lib, logger=log, out_dir=out_dir)
            
        log.info(f'finished \'{method}\' in {tdelta} on\n    {ofp}')
        
        return ofp, meta_lib
    
    def run_dsc_multi(self,
                      dem1_fp,wse2_fp,
                  
                  method_pars={'CostGrow': {}, 
                         'Basic': {}, 
                         'SimpleFilter': {}, 
                         'BufferGrowLoop': {}, 
                         'Schumann14': {},
                         },
                  write_meta=True,
                  **kwargs):
        """run downscaling on multiple methods
        
        Pars
        ------
        method_pars: dict
            method name: kwargs
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dscM', subdir=True, **kwargs)
        assert isinstance(method_pars, dict)
        assert set(method_pars.keys()).difference(self.nicknames_d.keys())==set()
        start = now()
        
        log.info(f'looping on {len(method_pars)}:\n    {list(method_pars.keys())}')
 
        #=======================================================================
        # precheck
        #=======================================================================
        assert_type_fp(dem1_fp, 'DEM')
        assert_type_fp(wse2_fp, 'WSE')
        assert_extent_equal(wse2_fp, dem1_fp), 'must pre-clip rasters'
        
        #=======================================================================
        # build common pars
        #=======================================================================
        downscale = self.get_downscale(wse2_fp, dem1_fp)
        
        log.info(f'downscale={downscale} on \n    WSE:{os.path.basename(wse2_fp)}\n    DEM:{os.path.basename(dem1_fp)}')
        
        #=======================================================================
        # loop on methods
        #=======================================================================
        res_lib = dict()
        for method, mkwargs in method_pars.items():

            d = dict()
            name = self.nicknames_d[method]
            skwargs = dict(out_dir=os.path.join(out_dir, name), 
                           logger=self.logger.getChild(name),
                           tmp_dir=os.path.join(tmp_dir, name),
                           )  
            log.info(f'on {name} w/ {mkwargs}\n\n')
            
 
            """not set up super well for sub-classing like this"""
            with Dsc_Session(obj_name=name,proj_name=self.proj_name,run_name=self.run_name,
                **skwargs) as wrkr:
                d['fp'], d['meta'] = wrkr.run_dsc(dem1_fp, wse2_fp, 
                                                  downscale=downscale,
                                                  subdir=False,
                                                  resname=self._get_resname(name), 
                                                  **skwargs)
                
            res_lib[method]=d
            
        log.info(f'finished on {len(res_lib)}\n\n')
        #=======================================================================
        # write meta summary
        #=======================================================================
        if write_meta:
            #collect
            meta_lib = {k:v['meta'].copy() for k,v in res_lib.items()}
            
            #===================================================================
            # with open(r'l:\09_REPOS\03_TOOLS\FloodDownscaler\coms\hp\tests\data\oop\run_dsc_multi_meta_lib_20230327.pkl',
            #           'wb') as file:
            #     pickle.dump(meta_lib, file)
            #===================================================================
 
            #convert
            self._write_meta(meta_lib)
            
            
            
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished in {now() - start}')
        
        return res_lib
        
        
                      
    
    #===========================================================================
    # PRIVATES--------
    #===========================================================================
 
def run_downscale(
        dem1_rlay_fp,
        wse2_rlay_fp,
        
        aoi_fp=None,
        method='CostGrow',
        **kwargs):
    """downscale/disag the wse (s2) raster to match the dem resolution (s1)
    
    Parameters
    ----------
    method: str
        downscaling method to apply. see run_dsc
        
    aoi_fp: str, Optional
        filepath to AOI. must be well rounded to the coarse raster
    """
    
    with Dsc_Session(aoi_fp=aoi_fp, **kwargs) as ses:
        wse1_dp_fp, meta_d = ses.run_dsc(dem1_rlay_fp, wse2_rlay_fp,  method=method)
        
    return wse1_dp_fp, meta_d

'''
Created on Dec. 4, 2022

@author: cefect

flood downscaling top-level control scripts
'''
import os, datetime, shutil
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

from fdsc.scripts.wbt import WBT_worker
from fdsc.base import (
    Master_Session, assert_dem_ar, assert_wse_ar, rlay_extract, nicknames_d, now
    )

from fdsc.scripts.simple import CostGrowSimple, BufferGrowLoop
from fdsc.scripts.schu14 import Schuman14





class Dsc_Session(CostGrowSimple, BufferGrowLoop, Schuman14,
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
        # get rlay write kwargs for this session
        # rlay_kwargs = get_write_kwargs(dem_stats, driver='GTiff', compress='LZW', masked=False)        
      
        self.s2, self.s1, self.downscale = s2, s1, downscale 
        return wse2_ar, dem1_ar, wse_stats, dem_stats
    


    #===========================================================================
    # PHASE1---------
    #===========================================================================


    #===========================================================================
    # PHASE2-----------------
    #===========================================================================
    def p2_dryPartials(self, wse1_fp, dem1_fp,
                       dryPartial_method='wetPartialsOnly',
                       write_meta=True,
                       run_kwargs=dict(),
                       **kwargs):
        """downscale in drypartial zones        
        should develop a few options here
        
        Parameters
        ----------
        dryPartial_method: str
            method to apply
            
        run_kwargs: dict
            pass kwargs to the run caller. used for testing.
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('p2DP', subdir=True, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        start = now()
        assert_spatial_equal(wse1_fp, dem1_fp)
        meta_lib = {'smry':{'dryPartial_method':dryPartial_method, 'wse1_fp':wse1_fp, 'dem1_fp':dem1_fp}}
            
        sn = nicknames_d[dryPartial_method]  # short name
        #=======================================================================
        # by method
        #=======================================================================
        if dryPartial_method == 'wetPartialsOnly':
            assert len(run_kwargs)==0
            rshutil.copy(wse1_fp, ofp, 'GTiff', strict=True, creation_options={})            
            wse1_dp_fp = ofp
            d = {'wetPartialsOnly':'none'}  # dummy placeholder
 
        elif dryPartial_method == 'costGrowSimple': 
            wse1_dp_fp, d = self.run_costGrowSimple(wse1_fp, dem1_fp, ofp=ofp, **run_kwargs, **skwargs)            
            
        elif dryPartial_method == 'bufferGrowLoop':
            wse1_dp_fp, d = self.run_bufferGrowLoop(wse1_fp, dem1_fp, ofp=ofp, **run_kwargs, **skwargs)            
            
        else:
            raise KeyError(dryPartial_method)
 
        meta_lib.update({sn + '_' + k:v for k, v in d.items()}) 
        #=======================================================================
        # check
        #=======================================================================
        if __debug__:
            assert_spatial_equal(wse1_fp, wse1_dp_fp)
            rlay_ar_apply(wse1_dp_fp, assert_wse_ar, masked=True)
        
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now() - start).total_seconds()
        meta_lib['smry']['tdelta'] = tdelta
        meta_lib['smry']['wse1_dp_fp'] = wse1_dp_fp
        log.info(f'finished in {tdelta:.2f} secs')
        
        if write_meta:
            self._write_meta(meta_lib, logger=log, out_dir=out_dir)
 
        return wse1_dp_fp, meta_lib

    def run_dsc(self,
            wse2_fp,
            dem1_fp,
 
            method='costGrowSimple',
            downscale=None,
            write_meta=True,
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
            
        Note
        -------
        no AOI clipping is performed. raster layers must have the same spatial extents. 
        see p0_clip_rasters to pre-clip the rasters
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dsc', subdir=True, **kwargs)
        meta_lib = {'smry':{**{'today':self.today_str, 'method':method}, **self._get_init_pars()}}
        skwargs = dict(logger=log, out_dir=out_dir, tmp_dir=tmp_dir)
        start = now()
        assert_extent_equal(wse2_fp, dem1_fp)
        
        
        #=======================================================================
        # precheck and load rasters
        #=======================================================================
 
        meta_lib['wse_raw'] = get_stats2(wse2_fp)
        meta_lib['dem_raw'] = get_stats2(dem1_fp)
        
        if downscale is None:
            downscale = self.get_downscale(wse2_fp, dem1_fp)
        
        #=======================================================================
        # run algo
        #=======================================================================
        if not method in ['schumann14', 'none']: #2 phase
            #=======================================================================
            # wet partials
            #=======================================================================                
            wse1_wp_fp, meta_lib['p1_wp'] = self.p1_wetPartials(wse2_fp, dem1_fp, 
                                                                downscale=downscale,
                                                                **skwargs)
     
            #=======================================================================
            # dry partials
            #=======================================================================
            wse1_dp_fp, meta_lib['p2_DP'] = self.p2_dryPartials(wse1_wp_fp, dem1_fp,
                                                    dryPartial_method=method,
                                                    **skwargs)
        
        elif method=='schumann14':
                wse1_dp_fp, md1 = self.run_schu14(wse2_fp, dem1_fp, downscale=downscale, **skwargs)
                
                meta_lib.update(md1)
                
        elif method=='none':            
            wse1_dp_fp, meta_lib['p1_wp'] = self.p1_wetPartials(wse2_fp, dem1_fp, 
                                                                downscale=downscale,
                                                                dem_filter=False,
                                                                **skwargs)
        
        else:
            raise KeyError(method)
        
        #=======================================================================
        # wrap
        #=======================================================================
        # copy tover to the main result
        rshutil.copy(wse1_dp_fp, ofp)
        
        tdelta = (now() - start).total_seconds()
        meta_lib['smry']['tdelta'] = tdelta
 
        if write_meta:
            self._write_meta(meta_lib, logger=log, out_dir=out_dir)
            
        log.info(f'finished on\n    {ofp}')
        
        return ofp, meta_lib
    
    #===========================================================================
    # PRIVATES--------
    #===========================================================================




def run_downscale(
        wse2_rlay_fp,
        dem1_rlay_fp,
        aoi_fp=None,
        method='costGrowSimple',
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
        wse1_dp_fp = ses.run_dsc(wse2_rlay_fp, dem1_rlay_fp, method=method)
        
    return wse1_dp_fp

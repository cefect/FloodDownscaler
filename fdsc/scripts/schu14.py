'''
Created on Jan. 25, 2023

@author: cefect

Scripts to replicate Schumann 2014's downscaling
'''


import os, datetime, shutil
import numpy as np
import rasterio as rio
from rasterio import shutil as rshutil

from fdsc.base import Dsc_basic, now

from hp.rio import (
     write_resample, assert_extent_equal, Resampling, assert_spatial_equal,
     write_mask_apply, 
     )

from hp.riom import (
    assert_mask_ar, load_mask_array, write_array_mask, write_extract_mask
    )


class Schuman14(Dsc_basic):
    
    
    def run_schu14(self,wse2_fp, dem_fp,
                   buffer_size=1.5,
                   gridcells=True,
 
                              **kwargs):
        """run python port of schuman 2014's downscaling
        
        Original script uses matlab's 'rangesearch' to match a 1.5 cell buffer
        
        
        Parameters
        ----------
        wse2_fp: str,
            filepath to coarse WSE (no phases)
            
        buffer_size: float, default 1.5
            size of buffer to include in downscale WSe search relative to the coarse resolution            
            see wbt.BufferRaster
            
        gridcells: bool, default False
            Optional flag to indicate that the 'size' threshold should be measured in grid cells instead of the default map units
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('schu14', subdir=False, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        assert_extent_equal(wse2_fp, dem_fp) 

        start = now()
        
        
        
        #get the downscale
        downscale = self.get_downscale(wse2_fp, dem_fp)
        
        meta_lib = {'smry':{
            'downscale':downscale, 'wse2_fp':os.path.basename(wse2_fp), 'dem_fp':dem_fp, 'ofp':ofp}}
        
        log.info(f'downscaling \'{os.path.basename(wse2_fp)}\' by {downscale} with buffer of {buffer_size}')
        #=======================================================================
        # get simple downscalled inundation
        #=======================================================================
        """ we want to allow buffer sizes as a fraction of the high-res grid
        """
        wse1_fp = write_resample(wse2_fp, scale=downscale, resampling=Resampling.nearest, out_dir=tmp_dir)
        assert_spatial_equal(dem_fp, wse1_fp)
        
        #=======================================================================
        # identify the search region
        #=======================================================================        
        buff_fp, meta_lib['buff'] = self.get_searchzone(wse1_fp, wbt_kwargs=dict(
            size=buffer_size*downscale, gridcells=gridcells), **skwargs)
        
        #=======================================================================
        # get the DEM within the search zone
        #=======================================================================
        buff_ar = load_mask_array(buff_fp, maskType='binary')
        demF_fp = write_mask_apply(dem_fp, buff_ar, logic=np.logical_or, ofp=os.path.join(tmp_dir, 'dem_masked.tif'))
        log.info(f'masked DEM to inundation search zone\n    {demF_fp}')
        
        #=======================================================================
        # filter to all those within buffer and less than DEM
        #=======================================================================
        wse2_fp, meta_lib['dscF'] = self.get_dsc_filtered(wse1_fp, dem_fp, buff_fp, **skwargs)
        
        #=======================================================================
        # wrap
        #=======================================================================
 
        rshutil.copy(wse2_fp, ofp)
        
 
        
    def get_dsc_filtered(self, wse1_fp, dem_fp, buff_fp, **kwargs):
        """
         filter to all those within buffer and less than DEM
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('buffer', subdir=False,  **kwargs)
        start = now()
        
        assert_spatial_equal(wse1_fp, dem_fp)
        assert_spatial_equal(buff_fp, dem_fp)
        
        
        #=======================================================================
        # mask wse
        #=======================================================================
        with rio.open(wse1_fp, mode='r') as wse_ds:
            wse_ar1 = wse_ds.read(1, masked=True)
            
            
            with rio.open(buff_fp, mode='r') as buff_ds:
                buff_ar = buff_ds.read(1, masked=False)
                
                raise NotImplementedError('not sure about this')
        
        
    def get_searchzone(self, wse1_fp,  
                   wbt_kwargs=dict(), **kwargs):
        """get a mask for the buffer search region"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('srch', subdir=False,  **kwargs)
 
        start = now()
        
        log.info('computing low-res buffer w/ %s'%wbt_kwargs)
        
 
        #=======================================================================
        # #convert inundation to mask
        #=======================================================================
        mask1_fp  = write_extract_mask(wse1_fp, out_dir=tmp_dir, maskType='binary')
        
        
        #=======================================================================
        # #make the buffer
        #=======================================================================
        buff1_fp = os.path.join(tmp_dir, 'buff1.tif')
        """this requires a binary type mask (just 1s and 0s"""
        assert self.buffer_raster(mask1_fp, buff1_fp, **wbt_kwargs)==0
        
        assert_spatial_equal(buff1_fp, wse1_fp)
        
        
        #=======================================================================
        # convert to donut
        #=======================================================================
        log.debug(f'computing donut on {buff1_fp}')
        #load the raw buffer
        buff1_ar = load_mask_array(buff1_fp, maskType='binary')
 
        
        #load the original mask
        mask1_ar = load_mask_array(mask1_fp, maskType='binary')
 
        #inside buffer but outside wse
        new_mask = np.logical_and(np.invert(buff1_ar), mask1_ar)
        
        #write the new mask
        write_array_mask(np.invert(new_mask), ofp=ofp, maskType='binary')
        
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now()-start).total_seconds()
        log.info(f'built buffer in {tdelta} w/ {new_mask.sum()}/{new_mask.size} active \n    {ofp}')
        
        return ofp, dict(tdelta=tdelta, ofp=ofp, mask_fp=mask1_fp, valid_cnt=new_mask.sum())
    
    
    
    
    
    
    
    
    
    
        
        
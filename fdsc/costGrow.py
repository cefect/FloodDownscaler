'''
Created on Mar. 2, 2023

@author: cefect

cost grow downscale algorthim
'''


import os, datetime, shutil
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
from rasterio import shutil as rshutil



#helpers
from hp.basic import now
from hp.gdal import getNoDataCount
from hp.rio import (
 
    write_array, assert_spatial_equal, get_write_kwargs, rlay_calc1, load_array, write_clip,
      get_ds_attr, get_stats2
    )

from hp.riom import write_extract_mask, write_array_mask, assert_mask_fp
from fdsc.base import assert_type_fp

#project
from fdsc.simple import WetPartials

from fdsc.base import (
    assert_dem_ar, assert_wse_ar, rlay_extract, assert_partial_wet
    )

class CostGrow(WetPartials):
    
    def __init__(self,
                 run_dsc_handle_d=None, 
                 **kwargs):
        
        if run_dsc_handle_d is None: run_dsc_handle_d=dict()
        
        run_dsc_handle_d['CostGrow'] = self.run_costGrow #add your main method to the caller dict
        
        super().__init__(run_dsc_handle_d=run_dsc_handle_d, **kwargs)
        
    def run_costGrow(self,wse_fp=None, dem_fp=None, 
                              **kwargs):
        """run CostGrow pipeline
        """
        method='CostGrow'
        log, tmp_dir, out_dir, ofp, resname = self._func_setup(self.nicknames_d[method],  **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        meta_lib=dict()
        assert_type_fp(wse_fp, 'WSE')
        assert_type_fp(dem_fp, 'DEM')
        
        #skwargs, meta_lib, log, ofp, start = self._func_setup_dsc(nicknames_d[method], wse_fp, dem_fp, **kwargs)
        downscale = self.downscale
        self._set_profile(dem_fp) #set raster profile
        #=======================================================================
        # p1: wet partials
        #=======================================================================                
        wse1_wp_fp, meta_lib['p1_wp'] = self.p1_wetPartials(wse_fp, dem_fp, downscale=downscale,**skwargs)
  
        #=======================================================================
        # p2: dry partials
        #=======================================================================
        wse1_dp_fp, meta_lib['p2_DP'] = self.get_costGrow_DP(wse1_wp_fp, dem_fp,ofp=ofp, **skwargs)     
 
        
        return wse1_dp_fp, meta_lib
        
        
        
    
    def get_costGrow_DP(self, wse_fp, dem_fp, **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('cg_dp', subdir=False, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        meta_lib={'smry':{'wse_fp':wse_fp}}
        start=now()
        assert_type_fp(wse_fp, 'WSE')
        self._set_profile(dem_fp) #set profile for session raster writing
        #=======================================================================
        # grow/buffer out the WSE values
        #=======================================================================
        costAlloc_fp, meta_lib['costDistanceGrow'] = self.get_costDistanceGrow_wbt(wse_fp, **skwargs)
 
        #=======================================================================
        # stamp out DEM violators
        #=======================================================================
        wse1_ar1_fp, meta_lib['filter_dem'] = self._filter_dem_violators(dem_fp, costAlloc_fp, **skwargs)
        
        #report
        if __debug__:
            og_noDataCount = getNoDataCount(wse_fp)
            new_noDataCount = meta_lib['filter_dem']['violation_count']
            assert og_noDataCount>0            
            
            assert   new_noDataCount<og_noDataCount
            
            log.info(f'dryPartial growth from {og_noDataCount} to {new_noDataCount} nulls '+\
                     f'({new_noDataCount/og_noDataCount:.2f})')
        
        #=======================================================================
        # remove isolated 
        #======================================================================= 
        wse1_ar2_fp, meta_lib['filter_iso'] = self._filter_isolated(wse1_ar1_fp,**skwargs)
        
        assert_spatial_equal(wse_fp, wse1_ar2_fp)
        #=======================================================================
        # wrap
        #=======================================================================
        rshutil.copy(wse1_ar2_fp, ofp)
        tdelta = (now()-start).total_seconds()
        meta_lib['smry']['tdelta'] = tdelta
        log.info(f'finished in {tdelta:.2f} secs')
        
        return ofp, meta_lib
    
    def _filter_dem_violators(self, dem_fp, wse_fp, **kwargs):
        """replace WSe values with nodata where they dont exceed the DEM"""
        #=======================================================================
        # defautls
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('filter', subdir=False,  **kwargs)
        assert_spatial_equal(dem_fp, wse_fp)
        
        """no... often we pass a costDistance raster which is WSE-like, but has no nulls
        assert_type_fp(wse_fp, 'WSE')"""
        #=======================================================================
        # load arrays
        #=======================================================================
        with rio.open( #load arrays
            wse_fp, mode='r') as ds:
            wse_ar = ds.read(1)
            assert not np.isnan(wse_ar).any(), 'shouldnt have any  nulls (we filled it!)'
            
        with rio.open(dem_fp, mode='r') as ds:
            dem1_ar = ds.read(1)
            
        #=======================================================================
        # #array math
        #=======================================================================
        bx_ar = wse_ar <= dem1_ar
        assert_partial_wet(bx_ar, msg='wse_ar <= dem1_ar')
 
        wse1_mar1 = ma.array(
            np.where(np.invert(bx_ar), wse_ar, np.nan),
            mask=bx_ar, fill_value=-9999)
        
        log.info(f'filtered {bx_ar.sum()}/{bx_ar.size} wse values which dont exceed the DEM')
        #=======================================================================
        # #dump to raster
        #=======================================================================
        #rlay_kwargs = get_write_kwargs(dem_fp, driver='GTiff', masked=False)
        wse1_ar1_fp = self.write_array(wse1_mar1, resname='wse1_ar3', 
                                       out_dir=out_dir,  logger=log, ofp=ofp) 
        
        
        #=======================================================================
        # meta
        #=======================================================================
        assert_type_fp(wse1_ar1_fp, 'WSE')
        
        meta_d={'size':wse_ar.size, 'wse1_ar1_fp':wse1_ar1_fp}
        if __debug__:
            meta_d['violation_count'] = bx_ar.astype(int).sum()
        
        
        return wse1_ar1_fp, meta_d

    def _filter_isolated(self, wse_fp, **kwargs):
        """remove isolated cells from grid using WBT"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('filter_iso', subdir=False,  **kwargs)
        start = now()
        meta_d=dict()
        assert get_ds_attr(wse_fp, 'nodata')==-9999
        assert_type_fp(wse_fp, 'WSE', msg='filter_iso input')
        #=======================================================================
        # #convert to mask
        #=======================================================================
        """wbt.clump needs 1s and 0s"""
        mask_fp = write_extract_mask(wse_fp, out_dir=out_dir, maskType='native')
        assert_mask_fp(mask_fp,  maskType='native')
        #=======================================================================
        # #clump it
        #=======================================================================
        clump_fp = os.path.join(tmp_dir, 'clump.tif')
        assert self.clump(mask_fp, clump_fp, diag=False, zero_back=True)==0
        meta_d['clump_fp'] = clump_fp
        meta_d['clump_mask_fp'] = mask_fp
        #=======================================================================
        # find main clump
        #=======================================================================
        with rio.open(clump_fp, mode='r') as ds:            
            mar = load_array(ds, masked=True)
            assert_partial_wet(mar.mask, 'expects some nodata cells on the clump result')
            ar = np.where(mar.mask, np.nan, mar.data)
            
            #identify the largest clump
            vals_ar, counts_ar = np.unique(ar, return_counts=True, equal_nan=True)
            
            assert len(vals_ar)>1, f'wbt.clump failed to identify enough clusters\n    {clump_fp}'
            
            max_clump_id = int(pd.Series(counts_ar, index=vals_ar).sort_values(ascending=False
                        ).reset_index().dropna('index').iloc[0, 0])
            
            #build a mask of this
            bx = ar==max_clump_id
            
            assert_partial_wet(bx)
            log.info(f'found main clump of {len(vals_ar)} with {bx.sum()}/{bx.size} unmasked cells'+\
                     '(%.2f)'%(bx.sum()/bx.size))
            
            meta_d.update({'clump_cnt':len(counts_ar), 'clump_max_size':bx.sum()})
            
        #=======================================================================
        # filter wse to main clump
        #=======================================================================
        with rio.open(wse_fp, mode='r') as ds:
            wse_ar = load_array(ds, masked=True)
            wse_ar1 = ma.array(wse_ar.data, mask = np.logical_or(
                np.invert(bx), #not in the clump
                wse_ar.mask, #dry
                ), fill_value=wse_ar.fill_value)
            
 
 
            profile = ds.profile
            
 
        assert_wse_ar(wse_ar1)
        meta_d.update({'raw_mask':wse_ar.mask.sum(), 'clump_filtered_mask':wse_ar1.mask.sum()})
        assert meta_d['raw_mask']<meta_d['clump_filtered_mask']
 
        #write
        write_array(wse_ar1, ofp=ofp, masked=False, **profile)
 
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now()-start).total_seconds()
        meta_d['tdelta'] = tdelta
        meta_d['ofp'] = ofp
        log.info(f'wrote {wse_ar1.shape} in {tdelta:.2f} secs to \n    {ofp}')
        
        return ofp, meta_d
    
    def get_costDistanceGrow_wbt(self, wse_fp,**kwargs):
        """cost grow/allocation using WBT"""
        start = now()
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('costGrow_wbt', subdir=False,  **kwargs)
        log.info(f'on {wse_fp}')
        meta_d = dict()
        #=======================================================================
        # costDistance
        #=======================================================================
        #fillnodata in wse (for source)
        wse_fp1 = os.path.join(tmp_dir, f'wse1_fnd.tif')
        assert self.convert_nodata_to_zero(wse_fp, wse_fp1) == 0
        
        #build cost friction (constant)
        cost_fric_fp = os.path.join(tmp_dir, f'cost_fric.tif')
        assert self.new_raster_from_base(wse_fp, cost_fric_fp, value=1.0, data_type='float') == 0
        meta_d['costFric_fp'] = cost_fric_fp
        
        #compute backlink raster
        backlink_fp = os.path.join(out_dir, f'backlink.tif')
        assert self.cost_distance(wse_fp1, 
            cost_fric_fp, 
            os.path.join(tmp_dir, f'backlink.tif'), backlink_fp) == 0
        
        meta_d['backlink_fp'] = backlink_fp
            
        log.info(f'built costDistance backlink raster \n    {backlink_fp}')
        
        #=======================================================================
        # costAllocation
        #=======================================================================
        costAlloc_fp = os.path.join(out_dir, 'costAllocation.tif')
        assert self.cost_allocation(wse_fp1, backlink_fp, costAlloc_fp) == 0
        meta_d['costAlloc_fp'] = costAlloc_fp
        #=======================================================================
        # wrap
        #=======================================================================
        
        
        assert_spatial_equal(costAlloc_fp, wse_fp)
        assert_type_fp(wse_fp, 'WSE')
        
        tdelta = (now() - start).total_seconds()
        meta_d['tdelta'] = tdelta
        log.info(f'finished in {tdelta}\n    {costAlloc_fp}')
        return costAlloc_fp, meta_d
    
 
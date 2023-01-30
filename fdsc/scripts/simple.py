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



from hp.gdal import getNoDataCount
from hp.rio import (
    assert_extent_equal, assert_ds_attribute_match, get_stats, assert_rlay_simple, RioSession,
    write_array, assert_spatial_equal, get_write_kwargs, rlay_calc1, load_array, write_clip,
    rlay_apply,rlay_ar_apply,write_resample, Resampling, get_ds_attr, get_stats2
    )

from fdsc.base import (
    Master_Session, assert_dem_ar, assert_wse_ar, rlay_extract, nicknames_d, now
    )

from fdsc.base import Dsc_basic



class WetPartials(Dsc_basic):
    """first phase of two phase downsamplers"""


    def p1_wetPartials(self, wse2_fp, dem_fp, downscale=None,
                       resampling=Resampling.bilinear,
                        **kwargs):
        """downscale wse2 grid in wet-partial regions
        
        Parameters
        ------------
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('p1WP', subdir=True, **kwargs)
        start = now()
        if downscale is None: 
            downscale = self.get_downscale(wse2_fp, dem_fp)

        log.info(f'downscale={downscale} on {os.path.basename(wse2_fp)} w/ {resampling}')
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
        
        meta_d['wse1_rsmp_fp'] = wse1_rsmp_fp
 
        #=======================================================================
        # #filter dem violators
        #=======================================================================
        wse1_filter_ofp, d = self.get_wse_dem_filter(wse1_rsmp_fp, dem_fp,logger=log, out_dir=tmp_dir)
        meta_d.update(d)
 
        #=======================================================================
        # wrap
        #=======================================================================
        rshutil.copy(wse1_filter_ofp, ofp)
        
        tdelta = (now() - start).total_seconds()
        meta_d['tdelta'] = tdelta
        
        log.info(f'built wse from downscale={downscale} on wet partials\n    {meta_d}\n    {ofp}')
        meta_d['wse1_wp_fp'] = ofp
        return ofp, meta_d

class TwoPhaseDSC(WetPartials):
    """methods common to simple two phase downsamplers"""
    
    def _func_setup_dsc(self, dkey, wse1_fp, dem_fp, **kwargs):
        log, tmp_dir, out_dir, ofp, resname = self._func_setup(dkey, subdir=False, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        assert_spatial_equal(dem_fp, wse1_fp)
        meta_lib = {'smry':{
            'wse1_fp':os.path.basename(wse1_fp), 'dem_fp':dem_fp, 'ofp':ofp}}
        start = now()
        return skwargs, meta_lib, log, ofp, start
    
class BufferGrowLoop(TwoPhaseDSC):
    def __init__(self, 
                 loop_range=range(30),
                 **kwargs):
        
        self.loop_range=loop_range
        
        super().__init__(**kwargs)
        
    def run_bufferGrowLoop(self,wse1_fp, dem_fp,
                       loop_range=None, 
                       min_growth_ratio=1.00001,
                              **kwargs):
        """loop of buffer + filter
        
        Parameters
        -------------
        loop_range: iterator, default range(30)
            buffer cell distance loop iterator. 
            
        min_growth_ratio: float, default 1.005
            minimum ratio of inundation count growth for buffer loop
            must be greater than 1.0
            values closer to 1.0 will allow the loop to continue
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if loop_range is None: loop_range=self.loop_range
        assert loop_range.__class__.__name__ == 'range'
        assert min_growth_ratio>=1.0
        
        skwargs, meta_lib, log, ofp, start = self._func_setup_dsc('bufg', wse1_fp, dem_fp, **kwargs)
        
        #=======================================================================
        # preload
        #=======================================================================
        dem_stats_d, dem_ar = rlay_extract(dem_fp)        
        assert_dem_ar(dem_ar)
        
        with rio.open(wse1_fp, mode='r') as ds:
            assert_rlay_simple(ds)
            wse1_ar = ds.read(1,  masked=True)
            prof = ds.profile
       
        assert_wse_ar(wse1_ar)
        
        assert not np.any(wse1_ar<dem_ar)
            
        
        #=======================================================================
        # buffer loop
        #=======================================================================
        log.info(f'on {loop_range}')
        wse1j_ar = np.where(wse1_ar.mask, np.nan, wse1_ar.data) #drop mask
        for i in loop_range:
            if i>min(wse1_ar.shape):
                log.warning(f'loop {i} exceeds minimum dimension of array.. breaking')
                break
            
            meta_d = {'pre_null_cnt':np.isnan(wse1j_ar).sum()}
            
            log.info(f'{i} w/ {meta_d}')
            #buffer
            wse1jb_ar = ar_buffer(wse1j_ar)
 
            
            #filter
            wse1j_ar = np.where(wse1jb_ar<=dem_ar.data, np.nan, wse1jb_ar)
            
            #wrap
            
            meta_d.update({'post_buff_null_cnt':np.isnan(wse1jb_ar).sum(),
                                      'post_filter_null_cnt':np.isnan(wse1j_ar).sum()})
            meta_d['growth_rate'] = meta_d['pre_null_cnt']/meta_d['post_filter_null_cnt']
            assert meta_d['growth_rate']>=1.0, 'lost inundation somehow...'
            
            if meta_d['growth_rate']<min_growth_ratio:
                log.warning(f'at i={i} growth_rate=%.2f failed to achieve minimum growth.. breaking'%(
                    meta_d['growth_rate']))
                break
            
            
            meta_lib[str(i)] = meta_d
            
        
 
        #=======================================================================
        # to raster
        #=======================================================================
        write_array(wse1j_ar, ofp, **prof) 
        
        log.info(f'wrote {wse1j_ar.shape} to \n    {ofp}')
 
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now()-start).total_seconds()
        meta_lib['smry']['tdelta'] = tdelta
        log.info(f'finished in {tdelta:.2f} secs')
        
        return ofp, meta_lib
 
class CostGrowSimple(TwoPhaseDSC):
    def run_costGrowSimple(self,wse1_fp, dem_fp, 
                              **kwargs):
        """dry partial algo with simple cost distancing
        
        #cost distance the voids (some distance constraint?)
        #filter dem violators
        #remove islands (wbt Clump?)
        #smooth
        """
        
        skwargs, meta_lib, log, ofp, start = self._func_setup_dsc('cgs', wse1_fp, dem_fp, **kwargs)
        #=======================================================================
        # grow/buffer out the WSE values
        #=======================================================================
        costAlloc_fp = self.get_costDistanceGrow_wbt(wse1_fp, **skwargs)
        meta_lib['smry']['costAlloc_fp'] = costAlloc_fp
        #=======================================================================
        # stamp out DEM violators
        #=======================================================================
        wse1_ar1_fp, meta_lib['filter_dem'] = self._filter_dem_violators(dem_fp, costAlloc_fp, **skwargs)
        
        #report
        if __debug__:

            og_noDataCount = getNoDataCount(wse1_fp)
            new_noDataCount = meta_lib['filter_dem']['violation_count']
            assert og_noDataCount>0            
            
            assert   new_noDataCount<og_noDataCount
            
            log.info(f'dryPartial growth from {og_noDataCount} to {new_noDataCount} nulls '+\
                     f'({new_noDataCount/og_noDataCount:.2f})')
        
        #=======================================================================
        # remove isolated 
        #======================================================================= 
        wse1_ar2_fp, meta_lib['filter_iso'] = self._filter_isolated(wse1_ar1_fp, ofp=ofp, **skwargs)
        
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now()-start).total_seconds()
        meta_lib['smry']['tdelta'] = tdelta
        log.info(f'finished in {tdelta:.2f} secs')
        
        return wse1_ar2_fp, meta_lib
    
    def _filter_dem_violators(self, dem_fp, wse_fp, **kwargs):
        """replace WSe values with nodata where they dont exceed the DEM"""
        #=======================================================================
        # defautls
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('filter', subdir=False,  **kwargs)
        assert_spatial_equal(dem_fp, wse_fp)
        
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
        wse1_ar1 = np.where(np.invert(bx_ar), wse_ar, np.nan)
        
        log.info(f'filtered {bx_ar.sum()}/{bx_ar.size} wse values which dont exceed the DEM')
        #=======================================================================
        # #dump to raster
        #=======================================================================
        rlay_kwargs = get_write_kwargs(dem_fp, driver='GTiff', masked=False)
        wse1_ar1_fp = self.write_array(wse1_ar1, resname='wse1_ar3', 
                                       out_dir=out_dir,  logger=log, ofp=ofp,
                                       **rlay_kwargs) 
        
        
        #=======================================================================
        # meta
        #=======================================================================
        meta_d={'size':wse_ar.size, 'wse1_ar1_fp':wse1_ar1_fp}
        if __debug__:
            meta_d['violation_count'] = bx_ar.astype(int).sum()
        
        
        return wse1_ar1_fp, meta_d

    def _filter_isolated(self, wse_fp, **kwargs):
        """remove isolated cells from grid using WBT"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('filter_iso', subdir=False,  **kwargs)
        start = now()
        meta_d=dict()
        #=======================================================================
        # #convert to mask
        #=======================================================================
        """not working
        self.raster_calculator(os.path.join(tmp_dir, 'rcalc.tif'),
                               statement="{raster}/{raster}".format(raster=f'\'{wse_fp}\''))"""
         
        mask_fp = rlay_calc1(wse_fp, os.path.join(tmp_dir, 'mask.tif'), lambda x:np.where(np.isnan(x), np.nan, 1.0))
        
        #=======================================================================
        # #clump it
        #=======================================================================
        clump_fp = os.path.join(tmp_dir, 'clump.tif')
        assert self.clump(mask_fp, clump_fp, diag=False, zero_back=True)==0
        meta_d['clump_fp'] = clump_fp
        #=======================================================================
        # find main clump
        #=======================================================================
        with rio.open(clump_fp, mode='r') as ds:            
            mar = load_array(ds, masked=True)
            ar = np.where(mar.mask, np.nan, mar.data)
            
            #identify the largest clump
            vals_ar, counts_ar = np.unique(ar, return_counts=True, equal_nan=True)
            
            max_clump_id = int(pd.Series(counts_ar, index=vals_ar).sort_values(ascending=False
                        ).reset_index().dropna('index').iloc[0, 0])
            
            #build a mask of this
            bx = ar==max_clump_id
            
            log.info(f'found main clump of {bx.sum()}/{bx.size} '+\
                     '(%.2f)'%(bx.sum()/bx.size))
            
            meta_d.update({'clump_cnt':len(counts_ar), 'clump_max_size':bx.sum()})
            
        #=======================================================================
        # filter wse to main clump
        #=======================================================================
        with rio.open(wse_fp, mode='r') as ds:
            ar = load_array(ds, masked=False)
            filtered_ar  = np.where(bx, ar, np.nan)
            profile = ds.profile
            
        #=======================================================================
        # #write
        #=======================================================================
        write_array(filtered_ar, ofp=ofp, masked=False, **profile)
 
            
        tdelta = (now()-start).total_seconds()
        meta_d['tdelta'] = tdelta
        meta_d['ofp'] = ofp
        log.info(f'wrote {filtered_ar.shape} in {tdelta:.2f} secs to \n    {ofp}')
        
        return ofp, meta_d
    
    def get_costDistanceGrow_wbt(self, wse_fp,**kwargs):
        """cost grow/allocation using WBT"""
        start = now()
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('costGrow_wbt', subdir=False,  **kwargs)
        log.info(f'on {wse_fp}')
        #=======================================================================
        # costDistance
        #=======================================================================
        #fillnodata in wse (for source)
        wse_fp1 = os.path.join(tmp_dir, f'wse1_fnd.tif')
        assert self.convert_nodata_to_zero(wse_fp, wse_fp1) == 0
        #build cost friction (constant)
        cost_fric_fp = os.path.join(tmp_dir, f'cost_fric.tif')
        assert self.new_raster_from_base(wse_fp, cost_fric_fp, value=1.0, data_type='float') == 0
        #compute backlink raster
        backlink_fp = os.path.join(out_dir, f'backlink.tif')
        assert self.cost_distance(wse_fp1, 
            cost_fric_fp,
 
            os.path.join(tmp_dir, f'backlink.tif'), backlink_fp) == 0
        log.info(f'built costDistance backlink raster \n    {backlink_fp}')
        
        #=======================================================================
        # costAllocation
        #=======================================================================
        costAlloc_fp = os.path.join(out_dir, 'costAllocation.tif')
        assert self.cost_allocation(wse_fp1, backlink_fp, costAlloc_fp) == 0
        log.info(f'finished in {now()-start}\n    {costAlloc_fp}')
        
        assert_spatial_equal(costAlloc_fp, wse_fp)
        return costAlloc_fp
    
 

def get_neighbours_D4(ar, mindex):
    """get values of d4 neighbours"""
    
    res_ar = np.full((3, 3), np.nan)
    val_d, loc_d = dict(), dict()
    for i, shift in enumerate([
        (0, 1), (0, -1),
        (1, 0), (-1, 0)]):
        
        # get shifted location
        jx = mindex[0] + shift[0]
        jy = mindex[1] + shift[1]
        
        if jx < 0 or jx >= ar.shape[0]:
            res = ma.masked
        elif jy < 0 or jy >= ar.shape[1]:
            res = ma.masked
        else:
            try:
                res = ar[jx, jy]
            except Exception as e:
                raise IndexError(
                    f'failed to retrieve value jy={jy} jx={jx} shape={ar.shape} w/ \n    {e}')
            
        res_ar[shift] = res
        
    #===========================================================================
    #     loc_d[i] = (jx, jy) 
    #     val_d[i]= res
    #     
    #     
    # print(mindex)
    # print(loc_d)
    # print(val_d)
    #===========================================================================
    
    return res_ar
 

def ar_buffer(wse_ar):
    """disaggregate/downscale the array to the specified scale
    
    results in an array with scale = ar.shape*downscale
    """
     
    res_ar = np.full(wse_ar.shape, np.nan)
    
    it = np.nditer([wse_ar, res_ar],
            flags=[
                'multi_index'
                # 'external_loop', 
                # 'buffered'
                ],
            op_flags=[['readonly'],
                        ['writeonly',
                         # 'allocate', #populate None 
                         # 'no_broadcast', #no aggregation?
                         ]],
            # op_axes=[None, new_shape],
            )
                         
    #===========================================================================
    # execute iteration
    #===========================================================================
    with it: 
        for wse, res in it:
            
            # dry
            if np.isnan(wse):
                # retrieve neighbours
                nei_ar = get_neighbours_D4(wse_ar, it.multi_index)
                
                # all dry
                if np.isnan(nei_ar).all():
                    res[...] = np.nan
                    
                # wet neighbours
                else:
                    res[...] = np.ravel(nei_ar[~np.isnan(nei_ar)]).mean()
                    #===========================================================
                    # #collapse to wet cells
                    # nei_ar2 = np.ravel(nei_ar[~np.isnan(nei_ar)])
                    # print(f'nei_ar2={nei_ar2}')
                    # 
                    # #higher than dem
                    # if nei_ar2.max()>dem:                    
                    #     res[...] = nei_ar2.max()
                    # else:
                    #     res[...] = np.nan
                    #===========================================================
 
            # wet
            else:
                res[...] = wse
            
            # print(f'{it.multi_index}: wse={wse} dem={dem}, res={res}')
                
        result = it.operands[-1]
        
    return result

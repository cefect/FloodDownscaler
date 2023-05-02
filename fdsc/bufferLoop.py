'''
Created on Mar. 2, 2023

@author: cefect
'''
import os, datetime, shutil
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
from rasterio import shutil as rshutil



from hp.basic import now
from hp.rio import (
    assert_rlay_simple, 
    write_array, 

    )

from fdsc.simple import WetPartials

from fdsc.base import (
    assert_dem_ar, assert_wse_ar, rlay_extract, assert_partial_wet
    )


class BufferGrowLoop(WetPartials):

    def __init__(self,
                 loop_range=range(30),
                  run_dsc_handle_d=None,
                 **kwargs):
        
        if run_dsc_handle_d is None: run_dsc_handle_d=dict()
        
        self.loop_range = loop_range
        
        run_dsc_handle_d['BufferGrowLoop'] = self.run_bufferGrowLoop  # add your main method to the caller dict
        
        super().__init__(run_dsc_handle_d=run_dsc_handle_d, **kwargs)
    
    def run_bufferGrowLoop(self, wse_fp=None, dem_fp=None,
                           loop_range=None,
                              **kwargs):
        """run CostGrow pipeline
        """
        method = 'BufferGrowLoop'
        log, tmp_dir, out_dir, ofp, resname = self._func_setup(self.nicknames_d[method], subdir=False, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        meta_lib = dict()
        
        # skwargs, meta_lib, log, ofp, start = self._func_setup_dsc(nicknames_d[method], wse_fp, dem_fp, **kwargs)
        downscale = self.downscale
        
        #=======================================================================
        # p1: wet partials
        #=======================================================================                
        wse1_wp_fp, meta_lib['p1_wp'] = self.p1_wetPartials(wse_fp, dem_fp, downscale=downscale, **skwargs)
  
        #=======================================================================
        # p2: dry partials
        #=======================================================================
        wse1_dp_fp, meta_lib['p2_DP'] = self.get_bufferGrowLoop_DP(wse1_wp_fp, dem_fp, ofp=ofp, 
                                                                   loop_range=loop_range, **skwargs)
        
        return wse1_dp_fp, meta_lib
        
    def get_bufferGrowLoop_DP(self, wse1_fp, dem_fp,
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
        if loop_range is None: loop_range = self.loop_range
        assert loop_range.__class__.__name__ == 'range'
        assert min_growth_ratio >= 1.0
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('bg_dp', subdir=False, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        meta_lib={'smry':{'wse_fp':wse1_fp}}
        start=now()
        
        #=======================================================================
        # preload
        #=======================================================================
        dem_stats_d, dem_ar = rlay_extract(dem_fp)        
        assert_dem_ar(dem_ar)
        
        with rio.open(wse1_fp, mode='r') as ds:
            assert_rlay_simple(ds)
            wse1_ar = ds.read(1, masked=True)
            prof = ds.profile
       
        assert_wse_ar(wse1_ar)
        
        assert not np.any(wse1_ar < dem_ar)
        
        #=======================================================================
        # buffer loop
        #=======================================================================
        log.info(f'on {loop_range}')
        wse1j_ar = np.where(wse1_ar.mask, np.nan, wse1_ar.data)  # drop mask
        for i in loop_range:
            if i > min(wse1_ar.shape):
                log.warning(f'loop {i} exceeds minimum dimension of array.. breaking')
                break
            
            meta_d = {'pre_null_cnt':np.isnan(wse1j_ar).sum()}
            
            log.info(f'{i} w/ {meta_d}')
            # buffer
            wse1jb_ar = ar_buffer(wse1j_ar)
            
            # filter
            wse1j_ar = np.where(wse1jb_ar <= dem_ar.data, np.nan, wse1jb_ar)
            
            # wrap
            
            meta_d.update({'post_buff_null_cnt':np.isnan(wse1jb_ar).sum(),
                                      'post_filter_null_cnt':np.isnan(wse1j_ar).sum()})
            meta_d['growth_rate'] = meta_d['pre_null_cnt'] / meta_d['post_filter_null_cnt']
            assert meta_d['growth_rate'] >= 1.0, 'lost inundation somehow...'
            
            if meta_d['growth_rate'] < min_growth_ratio:
                log.warning(f'at i={i} growth_rate=%.2f failed to achieve minimum growth.. breaking' % (
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
        tdelta = (now() - start).total_seconds()
        meta_lib['smry']['tdelta'] = tdelta
        log.info(f'finished in {tdelta:.2f} secs')
        
        return ofp, meta_lib

 
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

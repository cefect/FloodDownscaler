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
from hp.riom import write_extract_mask, write_array_mask, assert_mask

from fdsc.base import (
    Master_Session, assert_dem_ar, assert_wse_ar, rlay_extract, nicknames_d, now, assert_partial_wet
    )

from fdsc.base import Dsc_basic



class WetPartials(Dsc_basic):
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
        
        rlay_ar_apply(wse1_rsmp_fp, assert_wse_ar, msg='WSE resample')
        
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
                 run_dsc_handle_d=dict(), 
                 **kwargs):
        
        run_dsc_handle_d['Basic'] = self.run_basicDSC #add your main method to the caller dict
        
        super().__init__(run_dsc_handle_d=run_dsc_handle_d, **kwargs)
        
    def run_basicDSC(self,wse_fp=None, dem_fp=None, 
                              **kwargs):
        """run Basic (resample only) pipeline
        """
        method='Basic'
        log, tmp_dir, out_dir, ofp, resname = self._func_setup(nicknames_d[method], subdir=False, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir, ofp=ofp)
        meta_lib=dict()
        downscale = self.downscale
        
        
        wse1_wp_fp, meta_lib['p1_wp'] = self.p1_wetPartials(wse_fp, dem_fp, downscale=downscale,**skwargs)
        
        
        return wse1_wp_fp, meta_lib
        
        
class TwoPhaseDSC(WetPartials):
    """methods common to simple two phase downsamplers"""
    
    def _func_setup_dsc(self, dkey, wse1_fp, dem_fp, **kwargs):
        log, tmp_dir, out_dir, ofp, resname = self._func_setup(dkey, subdir=False, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        assert_spatial_equal(dem_fp, wse1_fp)
        meta_lib = {'smry':{'wse1_fp':os.path.basename(wse1_fp), 'dem_fp':dem_fp, 'ofp':ofp}}
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

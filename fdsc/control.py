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
from rasterio.enums import Resampling, Compression

import shapely.geometry as sgeo

from hp.basic import dstr, now
from hp.oop import Session
from hp.rio import (
    assert_extent_equal,  RioSession,get_profile,
    write_array, assert_spatial_equal, write_clip,get_meta, get_bbox,
    write_resample, get_meta
    )
from hp.pd import view
 
from hp.hyd import (
    HydTypes
    )

from fdsc.wbt import WBT_worker
from fdsc.base import (
    assert_dem_ar, assert_wse_ar, rlay_extract, assert_dsc_res_lib
    )

from fdsc.simple import BasicDSC
from fdsc.schu14 import Schuman14
from fdsc.costGrow import CostGrow
from fdsc.bufferLoop import BufferGrowLoop







class Dsc_Session_skinny(CostGrow, BufferGrowLoop, Schuman14,BasicDSC,WBT_worker):
    """workaround to avoid subclass conflicts"""
    
    def __init__(self, 
                 run_name='v1', #using v instead of r to avoid resolution confusion
                 relative=True,
                 **kwargs):
 
        super().__init__(run_name=run_name, relative=relative, **kwargs)
      
    #===========================================================================
    # phase0-------  
    #===========================================================================
    def p0_clip_rasters(self, dem_fp, wse_fp, 
                        clip_kwargs=dict(),
 
                        **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('clip', **kwargs)
        
        #ensure the profile is set from the dem
        self._set_profile(rlay_ref_fp=dem_fp)
 
        #=======================================================================
        # precheck
        #=======================================================================

 
        
 
        #=======================================================================
        # clip wse
        #=======================================================================
        #clip coarse WSE by aoi
        wse_clip_fp = self.clip_rlay(wse_fp, 
             clip_kwargs={**clip_kwargs, **dict(fancy_window=dict(round_offsets=True, round_lengths=True))},
                                            ofp=os.path.join(out_dir, 'wse2_clip.tif'),
                                            ) 
 
        #make this the new aoi
        self.bbox = get_bbox(wse_clip_fp)
        #round_aoi_fp = self.write_bbox_vlay(get_bbox(wse_clip_fp), out_dir=out_dir)
        
        
        #=======================================================================
        # clip DEM
        #=======================================================================
        dem_clip_fp = self.clip_rlay(dem_fp, bbox=self.bbox, ofp=os.path.join(out_dir, 'dem1_clip.tif'))
 
        #=======================================================================
        # warp
        #=======================================================================
        assert_extent_equal(wse_clip_fp, dem_clip_fp), 'must pre-clip rasters'
        HydTypes('DEM').assert_fp(dem_clip_fp)
        HydTypes('WSE').assert_fp(wse_clip_fp)
 
        log.info(f'clipped rasters and wrote to\n    {out_dir}\n    {self.bbox.bounds}')
        return dem_clip_fp, wse_clip_fp 
        
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
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dsc',  **kwargs)
        if debug is None: debug=__debug__
        
        
        meta_lib = {'smry':{**{'today':self.today_str, 'method':method, 
                               'wse2_fp':os.path.basename(wse2_fp), 
                               'dem_fp':dem1_fp, 'ofp':ofp}, 
                            **self._get_init_pars(), #wont have aoi props as these are on the calling session
                            }}
        
        skwargs = dict(logger=log, out_dir=out_dir, tmp_dir=tmp_dir)
        start = now()
 
        #=======================================================================
        # precheck and load rasters
        #=======================================================================
        #assert not os.path.exists(ofp), f'output exists\n    {ofp}'
        if debug:
            assert_extent_equal(wse2_fp, dem1_fp)
            HydTypes('DEM').assert_fp(dem1_fp)
            HydTypes('WSE').assert_fp(wse2_fp)
 
 
        meta_lib['wse_raw'] = get_meta(wse2_fp)
        meta_lib['dem_raw'] = get_meta(dem1_fp)
        
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
        HydTypes('WSE').assert_fp(wse1_fp) 
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
        
        return {'WSE1':ofp}, meta_lib
    
    def run_dsc_multi(self,
                      dem1_fp,wse2_fp,
                  
                  method_pars={
                        #'CostGrow': {}, 
                        'Basic': {}, 
                        'SimpleFilter': {}, 
                        #'BufferGrowLoop': {}, 
                        #'Schumann14': {},
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
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dscM', ext='.xls', **kwargs)
        assert isinstance(method_pars, dict)
        assert set(method_pars.keys()).difference(self.nicknames_d.keys())==set()
        start = now()
        
        log.info(f'looping on {len(method_pars)}:\n    {list(method_pars.keys())}')
 
        #=======================================================================
        # precheck
        #=======================================================================
        HydTypes('DEM').assert_fp(dem1_fp) 
        HydTypes('WSE').assert_fp(wse2_fp) 
 
        assert_extent_equal(wse2_fp, dem1_fp), 'must pre-clip rasters'
        
        #=======================================================================
        # build common pars
        #=======================================================================
        downscale = self.get_downscale(wse2_fp, dem1_fp)
        
        log.info(f'downscale={downscale} on \n    WSE:{os.path.basename(wse2_fp)}\n    DEM:{os.path.basename(dem1_fp)}')
        
        #=======================================================================
        # prep inputs
        #=======================================================================
        ins_d = {'DEM':dem1_fp, 'WSE2':wse2_fp}
 
        #=======================================================================
        # loop on methods
        #=======================================================================
        res_lib = {'inputs':{'fp':ins_d, 'meta':{'start':start}}}
        for method, mkwargs in method_pars.items():

            d = dict()
            name = self.nicknames_d[method]
 
            log.info(f'on {name} w/ {mkwargs}\n\n')
            
            d['fp'], d['meta'] = self.run_dsc(dem1_fp, wse2_fp, 
                                                  downscale=downscale,
                                                  resname=self._get_resname(name),
                                                  out_dir=os.path.join(out_dir, method),
                                                  write_meta=True, 
                                                  method=method,
                                                  rkwargs=mkwargs)
                
            res_lib[method]=d
            
        log.info(f'finished on {len(res_lib)}\n\n')
        """
        print(dstr(res_lib))
        """
        
        #=======================================================================
        # add relative paths
        #=======================================================================
        for k0, d0 in {k:v['fp'] for k,v in res_lib.items()}.items():
            res_lib[k0]['fp_rel'] = {k0:self._relpath(fp) for k0, fp in d0.items()}
 
        
        #=======================================================================
        # write meta summary
        #=======================================================================
        if write_meta:
            #collect
            meta_lib = {k:v['meta'].copy() for k,v in res_lib.items()}
 
            #convert
            self._write_meta(meta_lib, ofp=ofp)
            
 
        #=======================================================================
        # wrap
        #=======================================================================
        assert_dsc_res_lib(res_lib)
        log.info(f'finished in {now() - start}')
        
        return res_lib
    

    def build_agg_dem(self, dem1_fp, rescale_l, **kwargs):
        """build a set of new target (hi-res) DEM1
        
        Parameters
        ----------
        dsc_l: list
            downscales RELATIVE to WSE2
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, _, resname = self._func_setup('dem_rsmp', ext='.tif', subdir=True,
                                                               **kwargs)
        
        log.info(f'building {len(rescale_l)} DEMs from {os.path.basename(dem1_fp)}\n\n')
        dem_fp_d, meta_d = dict(), dict()
 
        #=======================================================================
        # loop each scale
        #=======================================================================
        for i, scale in enumerate(rescale_l):
            log.info(f'({i+1}/{len(rescale_l)}) w/ scale={scale}\n')
            rsmp_fp = os.path.join(out_dir, f'DEM_rsmp_d{scale*10:.0f}.tif')
            
            #===================================================================
            # build
            #===================================================================
            #take original
            if scale == 1:
                rshutil.copy(dem1_fp, rsmp_fp)
 
            else:
                _ = write_resample(dem1_fp, resampling=Resampling.average, 
                    scale=1/scale, ofp=rsmp_fp) #build a new one
                
            #===================================================================
            # #checks
            #===================================================================
            meta_d[scale]=get_meta(rsmp_fp)
            HydTypes('DEM').assert_fp(rsmp_fp)
            assert_extent_equal(rsmp_fp, dem1_fp), 'resampling issue'
            
            dem_fp_d[scale] = rsmp_fp
            
            log.debug(f'finished scale={scale} w/\n    {meta_d[scale]}')
                
        log.info(f'finished on {len(dem_fp_d)} to \n    {out_dir}')
        
        return dem_fp_d

    def run_dsc_multi_mRes(self,
                      wse2_fp, dem_fp_d,
                  
                  method_pars={
                        #'CostGrow': {}, 
                        'Basic': {}, 
                        'SimpleFilter': {}, 
                        #'BufferGrowLoop': {}, 
                        #'Schumann14': {},
                         },
 
                  write_meta=True,
 
                  **kwargs):
        """run downscaling on multiple methods and multi downscales
        
        Pars
        ------
        dsc_l: list
            downscales to build
            
        method_pars: dict
            method name: kwargs
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dscMd', ext='.xls', **kwargs)
        assert isinstance(method_pars, dict)
        assert set(method_pars.keys()).difference(self.nicknames_d.keys())==set()
        start = now()
        
        log.info(f'looping on {len(method_pars)}x{len(dem_fp_d)}:\n    {list(method_pars.keys())}')
 
        #=======================================================================
        # precheck
        #=======================================================================
         
        HydTypes('WSE').assert_fp(wse2_fp) 
 
 
        
        #=======================================================================
        # build common pars
        #=======================================================================
        def dskey(downscale):
            """converting downscale to more readble key"""
            return f'd{downscale*10:.0f}'
 
        log.info(f'on \n    WSE:{os.path.basename(wse2_fp)}')
        
        #re-key dems
        dsc_d = dict()
        for dem_scale, fp in dem_fp_d.items():
            dsc_d[dem_scale]=self.get_resolution_ratio(wse2_fp, fp)
        
        #=======================================================================
        # prep inputs
        #=======================================================================
        ins_d = {'WSE2':wse2_fp}
        ins_d.update({dskey(dsc_d[k]):v for k,v in dem_fp_d.items()})
        

        meta_d = {'start':start, 'dem_fp_d':dem_fp_d.copy(), 'dsc_d':dsc_d,
                  'dscale_nm_d':{dskey(v):v for k,v in dsc_d.items()},
                  }
        
 
        #=======================================================================
        # loop on methods
        #=======================================================================
        res_lib = {'inputs':{'inputs1':{'fp':ins_d, 
                                        'fp_rel':{k0:self._relpath(fp) for k0, fp in ins_d.items()},
                                        'meta':meta_d}}}        
        
        for method, mkwargs in method_pars.items(): 
            name = self.nicknames_d[method]
            log.info(f'{method} {name} w/ {mkwargs}----------\n\n')
            res_d=dict()
            
            #===================================================================
            # loop on scales
            #===================================================================
            for i, (dem_scale, dem1_fp) in enumerate(dem_fp_d.items()):
                #===============================================================
                # setup
                #===============================================================
 
                downscale=dsc_d[dem_scale] #WSE2 to WSE1
                dsc_str = dskey(downscale)
                resname_i=self._get_resname(f'{name}_{dsc_str}')
                odi=os.path.join(out_dir, method, dsc_str)
                if not os.path.exists(odi):os.makedirs(odi)
 
                log.info(f'({i+1}/{len(dem_fp_d)}) on {name} w/ downscale={downscale}\n    {dem1_fp}')
                
                #===============================================================
                # precheck
                #===============================================================
                HydTypes('DEM').assert_fp(dem1_fp)
                assert_extent_equal(wse2_fp, dem1_fp)
                
                #===============================================================
                # #build downscaled WSE2
                #===============================================================
                d=dict()
                if downscale!=1.0: 
                    d['fp'], d['meta'] = self.run_dsc(dem1_fp, wse2_fp, 
                                              downscale=downscale,resname=resname_i,out_dir=odi,
                                              write_meta=True, 
                                              method=method,
                                              rkwargs=mkwargs)
                else:
                    #no downscaling, just copy
                    ofpi = os.path.join(odi, f'wse2_copy_{method}_{dsc_str}.tif')
                    rshutil.copy(wse2_fp, ofpi)
                    d['fp']={'WSE1':ofpi}
                    d['meta'] ={'dem_raw':get_meta(dem1_fp)} #needed later
 
                #===============================================================
                # post
                #===============================================================
                #add relative filepaths
                d['fp_rel'] = {k0:self._relpath(fp) for k0, fp in d['fp'].items()}
                
                #store
                res_d[dsc_str]=d
                    
            res_lib[method]=res_d
            log.debug(f'finished {method}')
            
        log.info(f'finished on {len(res_lib)}\n\n')
 
        #=======================================================================
        # write meta summary
        #=======================================================================
        if write_meta:
            #collect
            meta_lib=dict()
            for k0, d0 in res_lib.items():
                #meta_lib[k0]=dict()
                for k1, d1 in d0.items():
                    meta_lib[k0+k1]=d1['meta']
                    
            #meta_lib = {k:v['meta'].copy() for k,v in res_lib.items()}
 
            #write
            self._write_meta(meta_lib, ofp=ofp)
            
 
        #=======================================================================
        # wrap
        #=======================================================================
 
        assert_dsc_res_lib(res_lib, level=2) 
        log.info(f'finished in {now() - start}')
        """
        print(res_lib['CostGrow'].keys())
        """
        
        return res_lib
 
 
class Dsc_Session(Session, RioSession, Dsc_Session_skinny):
    """session controller for downscaling"""

'''
Created on Dec. 4, 2022

@author: cefect
'''
import os, datetime, shutil
import numpy as np
import scipy
from pyproj.crs import CRS
import shapely.geometry as sgeo
import rasterio as rio
from rasterio import shutil as rshutil
import geopandas as gpd
import fiona
import numpy.ma as ma

 
def now():
    return datetime.datetime.now()



from hp.rio import (
    assert_extent_equal, assert_ds_attribute_match, get_stats, assert_rlay_simple, RioSession,
    write_array, assert_spatial_equal, get_write_kwargs, rlay_calc1, load_array, write_clip,
    rlay_apply,rlay_ar_apply,
    )
from hp.pd import view, pd
from hp.gdal import getNoDataCount

from fdsc.scripts.wbt import WBT_worker
from fdsc.scripts.coms2 import (
    Master_Session, assert_dem_ar, assert_wse_ar
    )
    


class Dsc_Session(RioSession,  Master_Session, WBT_worker):
    

      
    #===========================================================================
    # phase0-------  
    #===========================================================================
    def p0_clip_rasters(self, wse_fp, dem_fp, 
                        bbox=None,crs=None, 
                        **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('clip_rasters',  **kwargs) 
     
        write_kwargs =RioSession._get_defaults(self, bbox=bbox, crs=crs, as_dict=True)
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
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('load_rasters',  **kwargs)
        crs, bbox, compress, nodata =RioSession._get_defaults(self, crs=crs)
        

        #=======================================================================
        # load
        #=======================================================================
        #load wse with aoi rounded
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
            
            log.info('set crs from dem (%s)'%crs.to_epsg())
            
        assert dem_stats['crs']==crs, f'DEM crs %s doesnt match session {crs}'%dem_stats['crs']
        for stat in ['crs', 'bounds']:
            assert dem_stats[stat] == wse_stats[stat]
 
        assert s2 < s1, 'dem must have a finer resolution than the wse'
        if not s1 % s2 == 0.0:
            log.warning(f'uneven resolution relation ({s1}/{s2}={s1%s2})')
            
        #report
        downscale = s1 / s2
        log.info(f'downscaling from {s2} to {s1} ({downscale})')
        
 
        #=======================================================================
        # wrap
        #=======================================================================
        #get rlay write kwargs for this session
        #rlay_kwargs = get_write_kwargs(dem_stats, driver='GTiff', compress='LZW', masked=False)        
      
        self.s2, self.s1, self.downscale = s2, s1, downscale 
        return wse2_ar, dem1_ar, wse_stats, dem_stats 

    #===========================================================================
    # PHASE1---------
    #===========================================================================
    def p1_wetPartials(self, wse2_ar, dem1_ar,  downscale=None,**kwargs):
        """downscale wse2 grid in wet-partial regions"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('p1WP',subdir=True,  **kwargs)
        if downscale is None: downscale=self.downscale

        
        #=======================================================================
        # #precheck
        #=======================================================================
        assert_dem_ar(dem1_ar)
        assert_wse_ar(wse2_ar)
        
        for ds1, ds2 in zip(dem1_ar.shape, wse2_ar.shape):
            assert ds1/ds2==downscale, downscale
            
        #meta
        meta_d = {'downscale':downscale, 'wse2_shape':str(wse2_ar.shape)}
        
        def fmeta(ar, pfx): #meta updater
            meta_d.update({f'{pfx}_size':ar.size, f'{pfx}_nullCnt':np.isnan(ar).sum()})
        
        #=======================================================================
        # convert to nulls
        #=======================================================================
 
        wse2_arN = np.where(~wse2_ar.mask,wse2_ar.data,  np.nan)
        fmeta(wse2_arN, 'wse2')
        #=======================================================================
        # #simple zoom
        #=======================================================================
        wse1_ar1N = scipy.ndimage.zoom(wse2_arN,downscale, order=0, mode='reflect',   grid_mode=True)
        
        assert wse1_ar1N.shape == dem1_ar.shape
        
        fmeta(wse1_ar1N, 'wse1Z')
        #=======================================================================
        # #filter dem violators
        #=======================================================================
        wse_wp_bx = wse1_ar1N <= dem1_ar
        wse1_ar2N = np.where(np.invert(wse_wp_bx), wse1_ar1N, np.nan)
        
        fmeta(wse1_ar2N, 'wse1Zf')
        
        #=======================================================================
        # convert back to masked
        #=======================================================================
        wse1_ar2 = ma.array(wse1_ar2N, mask=np.isnan(wse1_ar2N), fill_value=wse2_ar.fill_value)
        
        #=======================================================================
        # wrap
        #=======================================================================
        assert_wse_ar(wse1_ar2)
        log.info(f'built wse from downscale={downscale} on wet partials w/ {wse_wp_bx.sum()}/{wse1_ar2.size} violators\n    {meta_d}')
        return wse1_ar2, meta_d

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

    #===========================================================================
    # PHASE2-----------------
    #===========================================================================
    def p2_dryPartials(self, wse1_fp, dem1_fp, 
                       dryPartial_method='none',
                       **kwargs):
        """downscale in drypartial zones        
        should develop a few options here
        
        Parameters
        ----------
        dryPartial_method: str
            method to apply
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('p2DP',subdir=True,  **kwargs)
        skwargs = dict(logger=log, out_dir=out_dir, tmp_dir=tmp_dir)
        start = now()
        assert_spatial_equal(wse1_fp, dem1_fp)
        meta_lib={'gen':{'dryPartial_method':dryPartial_method}}
        
 
            
        #=======================================================================
        # by method
        #=======================================================================
        if dryPartial_method == 'none':
            rshutil.copy(wse1_fp, ofp, 'GTiff', strict=True, creation_options={})            
            wse1_dp_fp=ofp
            
            """
            load_array(wse1_dp_fp, masked=True)
            load_array(wse1_fp, masked=True)
            """
 
 
        elif dryPartial_method == 'costGrowSimple': 
            wse1_dp_fp, d = self.p2_dp_costGrowSimple(wse1_fp, dem1_fp,**skwargs)
            
            meta_lib.update({'cgs_'+k:v for k,v in d.items()}) #append
 
            
        else:
            raise KeyError(dryPartial_method)
        """option 0.... Schuman 2014"""
        #buffer fixed number of pixels?
        """option3... buffer-filter loop. like costDistanceSimple but applies filter after each cell"""
        #for 1 cell
            #grow/buffer 1
            #filter dem violators
        """option 2... 1) identify hydraulic blocks; 2) apply 1D weighted smoothing""" 
        
        #=======================================================================
        # check
        #=======================================================================
        if __debug__:
            assert_spatial_equal(wse1_fp, wse1_dp_fp)
            rlay_ar_apply(wse1_dp_fp, assert_wse_ar, masked=True)
        
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now()-start).total_seconds()
        meta_lib['gen']['tdelta'] = tdelta
        log.info(f'finished in {tdelta:.2f} secs')
 
        return wse1_dp_fp, meta_lib



    def p2_dp_costGrowSimple(self,
                              wse2_fp, dem_fp, 
                              **kwargs):
        """dry partial algo with simple cost distancing
        
        #cost distance the voids (some distance constraint?)
        #filter dem violators
        #remove islands (wbt Clump?)
        #smooth
        """
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('cgs', subdir=False,  **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        assert_spatial_equal(dem_fp, wse2_fp)
        meta_lib = {'gen':{'wse2_fp':os.path.basename(wse2_fp)}}
        start = now()
        #=======================================================================
        # grow/buffer out the WSE values
        #=======================================================================
        costAlloc_fp = self.get_costDistanceGrow_wbt(wse2_fp, **skwargs)
        
        #=======================================================================
        # stamp out DEM violators
        #=======================================================================
        wse1_ar1_fp, meta_lib['filter_dem'] = self._filter_dem_violators(dem_fp, costAlloc_fp, **skwargs)
        
        #report
        if __debug__:

            og_noDataCount = getNoDataCount(wse2_fp)
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
        meta_lib['gen']['tdelta'] = tdelta
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
        meta_d={'size':wse_ar.size}
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
        log.info(f'wrote {filtered_ar.shape} in {tdelta:.2f} secs to \n    {ofp}')
        
        return ofp, meta_d
    
    #===========================================================================
    # PIPELINE
    #===========================================================================



    def run_dsc(self,
            wse2_rlay_fp,
            dem1_rlay_fp,
 
            dryPartial_method = 'costGrowSimple',
                **kwargs):
        """run a downsampling pipeline
        
        Paramerters
        -------------
        wse2_rlay_fp: str
            filepath to WSE raster layer at low-resolution (to be downscaled)
            
        dem1_rlay_fp: str
            filepath to DEM raster layer at high-resolution (used to infer downscaled WSE)
            
        Note
        -------
        no AOI clipping is performed. raster layers must have the same spatial extents. 
        see p0_clip_rasters to pre-clip the rasters
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dsc', subdir=False,  **kwargs)
        meta_lib = {'smry':{**{'today':self.today_str}, **self._get_init_pars()}}
        #=======================================================================
        # precheck and load rasters
        #=======================================================================        
        wse2_ar, dem1_ar, wse_stats, dem_stats  = self.p0_load_rasters(wse2_rlay_fp, dem1_rlay_fp, logger=log)
        
        
        
        #get default writing parmaeters
        rlay_kwargs = self._get_defaults(as_dict=True)        
        rlay_kwargs.update({'transform':dem_stats['transform'], 'dtype':'float32'})
        del rlay_kwargs['bbox']
        
        outres = dem_stats['res'][0]
        outName_sfx = f'r{outres:02.0f}'
        
        #update meta
        meta_lib['grid'] = rlay_kwargs
        meta_lib['wse_raw'], meta_lib['dem_raw'] = wse_stats, dem_stats
        
        #=======================================================================
        # wet partials
        #=======================================================================
        wse1_ar2, meta_lib['p1_wp'] = self.p1_wetPartials(wse2_ar, dem1_ar, logger=log)
        
        """
        np.save(r'l:\09_REPOS\03_TOOLS\FloodDownscaler\tests\data\fred01\wse1_ar2', wse1_ar2, fix_imports=False)
        """
        
        #convert back to raster
        wse1_wp_fp = self.write_array(wse1_ar2, resname='wse1_wp', out_dir=tmp_dir, masked=True, **rlay_kwargs)
        #=======================================================================
        # dry partials
        #=======================================================================
        wse1_dp_fp, meta_lib['p2_DP'] = self.p2_dryPartials(wse1_wp_fp, dem1_rlay_fp, dryPartial_method=dryPartial_method, 
                                         logger=log)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished on\n    {wse1_dp_fp}')
        
        return wse1_dp_fp
    
    #===========================================================================
    # PRIVATES--------
    #===========================================================================
    

 
 

def rlay_extract(fp,
                 window=None, masked=True,
 
                 ):
    
    if not masked:
        raise NotImplementedError(masked)
    
    """load rlay data and arrays"""
    with rio.open(fp, mode='r') as ds:
        assert_rlay_simple(ds)
        stats_d = get_stats(ds) 
 
        ar = ds.read(1, window=window, masked=masked)
        
        stats_d['null_cnt'] = ar.mask.sum()
        
    return stats_d, ar 


def run_downscale(
        wse2_rlay_fp,
        dem1_rlay_fp,
        aoi_fp=None, 
        dryPartial_method = 'costGrowSimple',
        **kwargs):
    """downscale/disag the wse (s2) raster to match the dem resolution (s1)
    
    Parameters
    ----------
    dryPartial_method: str
        dry partial algo method
        
    aoi_fp: str, Optional
        filepath to AOI. must be well rounded to the coarse raster
    """
    
    with Dsc_Session(aoi_fp=aoi_fp, **kwargs) as ses:
        wse1_dp_fp = ses.run_dsc(wse2_rlay_fp,dem1_rlay_fp,dryPartial_method=dryPartial_method)

        
    return wse1_dp_fp

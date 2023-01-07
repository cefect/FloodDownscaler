'''
Created on Dec. 4, 2022

@author: cefect
'''
import os, datetime
import numpy as np
import scipy
from pyproj.crs import CRS
import shapely.geometry as sgeo
import rasterio as rio
import geopandas as gpd
import fiona


 
def now():
    return datetime.datetime.now()



from hp.rio import (
    assert_extent_equal, assert_ds_attribute_match, get_stats, assert_rlay_simple, RioSession,
    write_array, assert_spatial_equal, get_write_kwargs, rlay_calc1, load_array, write_clip)
from hp.pd import view, pd
from hp.gdal import getNoDataCount

from fdsc.scripts.wbt import WBT_worker
from fdsc.scripts.coms2 import Master_Session
    


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
    def p1_downscale_wetPartials(self, wse2_ar, dem1_ar,  downscale=None, **kwargs):
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('zoom',  **kwargs)
        if downscale is None: downscale=self.downscale
        
        #precheck
        for ds1, ds2 in zip(dem1_ar.shape, wse2_ar.shape):
            assert ds1/ds2==downscale, downscale
        
        #simple zoom
        wse1_ar1 = scipy.ndimage.zoom(wse2_ar, downscale, order=0, mode='reflect',   grid_mode=True)
        assert wse1_ar1.shape == dem1_ar.shape
        
        #filter dem violators
        wse_wp_bx = wse1_ar1 <= dem1_ar
        wse1_ar2 = np.where(np.invert(wse_wp_bx), wse1_ar1, np.nan)
        log.info(f'built wse from downscale={downscale} on wet partials w/ {wse_wp_bx.sum()}/{wse1_ar2.size} violators')
        return wse1_ar2

    def get_costGrow_wbt(self, wse_fp,**kwargs):
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
        return costAlloc_fp

    #===========================================================================
    # PHASE2-----------------
    #===========================================================================

    def _filter_dem_violators(self, dem_fp, costAlloc_fp):
        with rio.open( #load arrays
            costAlloc_fp, mode='r') as ds:
            costAlloc_ar = ds.read(1)
            assert not np.isnan(costAlloc_ar).any(), 'shouldnt have any  nulls (we filled it!)'
        with rio.open(dem_fp, mode='r') as ds:
            dem1_ar = ds.read(1)
    #array math
        bx_ar = costAlloc_ar <= dem1_ar
        wse1_ar1 = np.where(np.invert(bx_ar), costAlloc_ar, np.nan)
        return bx_ar, wse1_ar1

    def p2_dp_costGrowSimple(self,
                              wse2_fp, dem_fp, 
                              **kwargs):
        """dry partial algo with simple cost distancing
        
        #cost distance the voids (some distance constraint?)
        #filter dem violators
        #remove islands (wbt Clump?)
        #smooth
        """
        start = now()
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dpCGS', subdir=True,  **kwargs)
        skwargs = dict(logger=log, out_dir=out_dir, tmp_dir=tmp_dir)
        assert_spatial_equal(dem_fp, wse2_fp)
        
        #=======================================================================
        # grow/buffer out the WSE values
        #=======================================================================
        costAlloc_fp = self.get_costGrow_wbt(wse2_fp, **skwargs)
        
        #=======================================================================
        # stamp out DEM violators
        #=======================================================================
        bx_ar, wse1_ar1 = self._filter_dem_violators(dem_fp, costAlloc_fp)
        
        #report
        if __debug__:
            og_noDataCount = getNoDataCount(wse2_fp)
            assert og_noDataCount>0
            
            new_noDataCount = bx_ar.astype(int).sum()
            
            assert new_noDataCount< og_noDataCount
            
            log.info(f'dryPartial growth from {og_noDataCount} to {new_noDataCount} nulls '+\
                     f'({new_noDataCount/og_noDataCount})')
        else:
            log.info(f'finished dryPartial growth on {wse1_ar1.shape}')        
        
        #=======================================================================
        # remove isolated 
        #=======================================================================
        #dump to raster
        rlay_kwargs = get_write_kwargs(dem_fp, driver='GTiff', masked=False)
        wse1_ar1_fp = self.write_array(wse1_ar1, resname='wse1_ar3', out_dir=tmp_dir,  logger=log, **rlay_kwargs) 
        
        #filter
        wse1_ar2_fp = self.filter_isolated(wse1_ar1_fp, ofp=ofp, **skwargs)
        
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now()-start).total_seconds()
        log.info(f'finished in {tdelta:.2f} secs')
        
        return wse1_ar2_fp

    def filter_isolated(self, wse_fp, **kwargs):
        """remove isolated cells from grid using WBT"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('filter_iso', subdir=False,  **kwargs)
        start = now()
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
        with rio.open(ofp, mode='w', **profile) as dest:
            dest.write(filtered_ar, 1)
            
        tdelta = (now()-start).total_seconds()
        log.info(f'wrote {filtered_ar.shape} in {tdelta:.2f} secs to \n    {ofp}')
        
        return ofp
    
    #===========================================================================
    # PRIVATES--------
    #===========================================================================
    #===========================================================================
    # def func_setup(self, *args, crs=None,bbox=None, **kwargs):
    #     """function setup wrapper"""
    #      
    #     if crs is None:
    #         crs=self.crs
    #          
    #     return crs, *self._func_setup(*args, **kwargs)
    #===========================================================================
    
 

def rlay_extract(fp,
                 window=None, masked=False,
 
                 ):
    
    """load rlay data and arrays"""
    with rio.open(fp, mode='r') as ds:
        assert_rlay_simple(ds)
        stats_d = get_stats(ds) 
 
        ar = ds.read(1, window=window, masked=masked)
        
    return stats_d, ar 


def run_downscale(
        wse2_rlay_fp,
        dem1_rlay_fp,
        aoi_fp=None,
 
        dryPartial_method = 'costDistanceSimple',
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
        log=ses.logger.getChild('m')
        #=======================================================================
        # precheck and load rasters
        #=======================================================================
        #trim rasters
        if not aoi_fp is None:
            wse2_rlay1_fp, dem1_rlay1_fp = ses.p0_clip_rasters(wse2_rlay_fp, dem1_rlay_fp)
        else:
            wse2_rlay1_fp, dem1_rlay1_fp = wse2_rlay_fp, dem1_rlay_fp
            
        
        
        wse2_ar, dem1_ar, wse_stats, dem_stats  = ses.p0_load_rasters(wse2_rlay1_fp, dem1_rlay1_fp)
        
        #get default writing parmaeters
        rlay_kwargs = ses._get_defaults(as_dict=True)        
        rlay_kwargs.update({'transform':dem_stats['transform'], 'dtype':'float32'})
        del rlay_kwargs['bbox']
        
        outres = dem_stats['res'][0]
        outName_sfx = f'r{outres:02.0f}'
        #=======================================================================
        # wet partials
        #=======================================================================
        wse1_ar2 = ses.p1_downscale_wetPartials(wse2_ar, dem1_ar)
        
        """
        np.save(r'l:\09_REPOS\03_TOOLS\FloodDownscaler\tests\data\fred01\wse1_ar2', wse1_ar2, fix_imports=False)
        """
        
        #=======================================================================
        # dry partials
        #=======================================================================
        """should develop a few options here"""
        
        if dryPartial_method=='none':
            wse1_dp_fp = ses.write_array(wse1_ar2, ofp=ses._get_ofp(dkey='dpNone_'+outName_sfx,  ext='.tif') ,  
                                         **rlay_kwargs)
            

        elif dryPartial_method=='costDistanceSimple':
 
            #convert back to rasters
            wse1_wp_fp = ses.write_array(wse1_ar2, resname='wse1_wp', out_dir=ses.tmp_dir,  **rlay_kwargs) 
            
            #grow out into dry partials
            wse1_dp_fp = ses.p2_dp_costGrowSimple(wse1_wp_fp, dem1_rlay1_fp, 
                                                  ofp=ses._get_ofp(dkey='cds_'+outName_sfx,  ext='.tif'))
            
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
        # wrap
        #=======================================================================
        log.info(f'finished on\n    {wse1_dp_fp}')
        
    return wse1_dp_fp

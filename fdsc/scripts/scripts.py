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


from hp.oop import Session
from hp.rio import (
    assert_extent_equal, assert_ds_attribute_match, get_stats, assert_rlay_simple, RioWrkr,
    write_array, assert_spatial_equal, get_write_kwargs, rlay_calc1, load_array)
from hp.pd import view, pd
from hp.gdal import getNoDataCount

from fdsc.scripts.wbt import WBT_worker


class Dsc_Session(RioWrkr,  Session, WBT_worker):
    
    def __init__(self, 
                 #==============================================================
                 # crs=CRS.from_user_input(25832),
                 # bbox=sgeo.box(0, 0, 100, 100),
                 #==============================================================
                 crs=None, bbox=None, aoi_fp=None,
                 
                 **kwargs):
        
        #=======================================================================
        # set aoi
        #=======================================================================
        if not aoi_fp is None:
            assert os.path.exists(aoi_fp)
            assert crs is None
            assert bbox is None
            
            #open file and get bounds and crs using fiona
            with fiona.open(aoi_fp, "r") as source:
                bbox = sgeo.box(*source.bounds) 
                crs = CRS(source.crs['init'])
            
 
        super().__init__(**kwargs)
        
        self.crs=crs
        self.bbox = bbox
      
    #===========================================================================
    # phase0-------  
    #===========================================================================
    def p0_load_rasters(self, wse2_rlay_fp, dem1_rlay_fp, 
                        bbox=None,   **kwargs):
        """load and extract some data from the raster files"""
        crs, log, tmp_dir, out_dir, ofp, resname = self._func_setup('load_rasters',  **kwargs)
        
        if bbox is None: bbox=self.bbox
        #if crs is None: crs=self.crs
        #=======================================================================
        # load
        #=======================================================================
        
        
        dem_stats, dem1_ar = rlay_extract(dem1_rlay_fp, bbox=bbox, crs=crs)
        wse_stats, wse2_ar = rlay_extract(wse2_rlay_fp, bbox=bbox, crs=crs)
        s2, s1 = dem_stats['res'][0], wse_stats['res'][0]
        
        #=======================================================================
        # check
        #=======================================================================
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
        rlay_kwargs = get_write_kwargs(dem_stats, driver='GTiff', compress='LZW', masked=False)
        
        #attach some session defaults
        for k in ['crs', 'nodata', 'transform']:
            setattr(self, k, dem_stats[k])
        
        self.s2, self.s1, self.downscale, self.rlay_kwargs = s2, s1, downscale, rlay_kwargs

        return dem1_ar, wse2_ar, rlay_kwargs

    #===========================================================================
    # PHASE1---------
    #===========================================================================
    def p1_downscale_wetPartials(self, wse2_ar, dem1_ar,  downscale=None, **kwargs):
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('zoom',  **kwargs)
        if downscale is None: downscale=self.downscale
        
        #precheck
        for ds1, ds2 in zip(dem1_ar.shape, wse2_ar.shape):
            assert ds1/ds2==downscale
        
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
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dp1', subdir=True,  **kwargs)
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
        
        wse1_ar2_fp = self.filter_isolated(wse1_ar1_fp, **skwargs)
        
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now()-start).total_seconds()
        log.info(f'finished in {tdelta:.2f}secs')
        
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
    def _func_setup(self, *args, crs=None, **kwargs):
        """function setup wrapper"""
        
        if crs is None:
            crs=self.crs
            
        return crs, *super(Dsc_Session, self)._func_setup(*args, **kwargs)
    
 

def rlay_extract(fp,
                 window=None, masked=False,
                 crs=None, bbox=None,
                 ):
    """load rlay data and arrays"""
    with rio.open(fp, mode='r') as ds:
        assert_rlay_simple(ds)
        stats_d = get_stats(ds)
        
        if not crs is None:
            assert crs==ds.crs
        
        #window
        if window is None and not bbox is None:
            #get a nice rounded window from the bbox
            window = rio.windows.from_bounds(*bbox.bounds, transform=ds.transform).round_offsets().round_lengths()
        
        ar = ds.read(1, window=window, masked=masked)
        
    return stats_d, ar


def run_downscale(
        wse2_rlay_fp,
        dem1_rlay_fp,
 
        dryPartial_method = 'costDistanceSimple',
        **kwargs):
    """downscale/disag the wse (s2) raster to match the dem resolution (s1)
    
    Parameters
    ----------
    dryPartial_method: str
        dry partial algo method
    """
    
    with Dsc_Session(**kwargs) as ses:
        log=ses.logger.getChild('m')
        #=======================================================================
        # precheck rasters
        #=======================================================================
        dem1_ar, wse2_ar, rlay_kwargs = ses.p0_load_rasters(wse2_rlay_fp, dem1_rlay_fp)
 
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
        
        """option 0.... Schuman 2014"""
        #buffer fixed number of pixels?
        
        """option 1... just cost distance"""

        if dryPartial_method=='costDistanceSimple':
            #convert back to rasters
            wse1_wp_fp = ses.write_array(wse1_ar2, resname='wse1_wp', out_dir=ses.tmp_dir,  **rlay_kwargs) 
            
            #grow out into dry partials
            wse1_dp_fp = ses.p2_dp_costGrowSimple(wse1_wp_fp, dem1_rlay_fp)
            
        else:
            raise KeyError(dryPartial_method)
        
        """option3... buffer-filter loop. like costDistanceSimple but applies filter after each cell"""
        #for 1 cell
            #grow/buffer 1
            #filter dem violators
        
        """option 2... 1) identify hydraulic blocks; 2) apply 1D weighted smoothing"""
        

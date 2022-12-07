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
 
def now():
    return datetime.datetime.now()

from hp.oop import Session
from hp.rio import (
    assert_extent_equal, assert_ds_attribute_match, get_stats, assert_rlay_simple, RioWrkr,
    write_array, assert_spatial_equal)

from fdsc.scripts.wbt import WBT_worker

class Dsc_Session(RioWrkr,  Session, WBT_worker):
    
    def __init__(self, 
                 #==============================================================
                 # crs=CRS.from_user_input(25832),
                 # bbox=sgeo.box(0, 0, 100, 100),
                 #==============================================================
                 **kwargs):
        
        super().__init__(**kwargs)
        
        #=======================================================================
        # self.crs=crs
        # self.bbox = bbox
        #=======================================================================
        
    def downscale_simple_zoom(self,
                              ar_raw,
                              downscale,
                              **kwargs):
        
        """just populates at finer resolution"""
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('zoom',  **kwargs)
        assert downscale>1
        assert len(ar_raw.shape)==2
        
        return scipy.ndimage.zoom(ar_raw, downscale, order=0, mode='reflect',   grid_mode=True)
    
    

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
    
    def null_dem_violators(self, dem_fp, wse_fp,
                           **kwargs):
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dem_filter', subdir=True,  **kwargs)

    def dp_costGrowSimple(self,
                              dem_fp, wse_fp,
                              **kwargs):
        """dry partial algo with simple cost distancing
        
        #cost distance the voids (some distance constraint?)
        #filter dem violators
        #remove islands (wbt Clump?)
        #smooth
        """
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dp1', subdir=True,  **kwargs)
        skwargs = dict(logger=log, out_dir=out_dir, tmp_dir=tmp_dir)
        assert_spatial_equal(dem_fp, wse_fp)
        
        #=======================================================================
        # grow/buffer out the WSE values
        #=======================================================================
        costAlloc_fp = self.get_costGrow_wbt(wse_fp, **skwargs)
        
        #=======================================================================
        # stamp out DEM violators
        #=======================================================================
        with rio.open(costAlloc_fp, mode='r') as ds:
            ar = ds.read(1)
        
        return costAlloc_fp
        
    


def rlay_extract(fp,
                 window=None, masked=False
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
        dem_stats, dem1_ar = rlay_extract(dem1_rlay_fp)
        wse_stats, wse2_ar = rlay_extract(wse2_rlay_fp)
        s2, s1 = dem_stats['res'][0], wse_stats['res'][0]
                    
        for stat in ['crs', 'bounds']:
            assert dem_stats[stat]==wse_stats[stat]
            
        #assert dem_stats['crs']==ses.crs
        
        assert s2<s1, 'dem must have a finer resolution than the wse'
        
        if not s1%s2==0.0:
            log.warning(f'uneven resolution relation ({s1}/{s2}={s1%s2})')
            
        downscale=s1/s2        
        
        log.info(f'downscaling from {s2} to {s1} ({downscale})')
        
        #attach some session defaults
        for k in ['crs', 'nodata', 'transform']:
            setattr(ses, k, dem_stats[k])
            
        #get rlay write kwargs for this session
        rlay_kwargs = ses._get_write_kwargs(dem_stats, driver='GTiff', compress='LZW', masked=False)
 
        #=======================================================================
        # wet partials
        #=======================================================================
        #simple zoom
        wse1_ar1 = ses.downscale_simple_zoom(wse2_ar, downscale)
        
        assert wse1_ar1.shape==dem1_ar.shape
        
        #filter dem violators
        wse_wp_bx = wse1_ar1<=dem1_ar
        wse1_ar2 = np.where(wse1_ar1>=dem1_ar, wse1_ar1, np.nan)
        
        log.info(f'built wse at {s1} on wet partials w/ {wse_wp_bx.sum()}/{wse1_ar2.size} violators')
        
        #=======================================================================
        # dry partials
        #=======================================================================
        """should develop a few options here"""
        
        """option 0.... Schuman 2014"""
        #buffer fixed number of pixels?
        
        """option 1... just cost distance"""

        if dryPartial_method=='costDistanceSimple':
            #convert back to rasters
            wse1_ar2_fp = ses.write_array(wse1_ar2, resname='wse1_ar2', out_dir=ses.tmp_dir,  **rlay_kwargs) 
            
            ses.dp_costGrowSimple(dem1_rlay_fp, wse1_ar2_fp)
            
        else:
            raise KeyError(dryPartial_method)
        
        
        """option 2... 1) identify hydraulic blocks; 2) apply 1D weighted smoothing"""
        
        
        
        
        
        
        
            
        
 
        
        
'''
Created on Dec. 4, 2022

@author: cefect
'''
import numpy as np
import scipy
from pyproj.crs import CRS
import shapely.geometry as sgeo
import rasterio as rio


from hp.oop import Session
from hp.rio import assert_extent_equal, assert_ds_attribute_match, get_stats, assert_rlay_simple

class Dsc_Session(Session):
    
    def __init__(self, 
                 crs=CRS.from_user_input(25832),
                 bbox=sgeo.box(0, 0, 100, 100),
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.crs=crs
        self.bbox = bbox
        
    def downscale_simple_zoom(self,
                              ar_raw,
                              downscale,
                              **kwargs):
        
        """just populates at finer resolution"""
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('zoom',  **kwargs)
        assert downscale>1
        assert len(ar_raw.shape)==2
        
        return scipy.ndimage.zoom(ar_raw, downscale, order=0, mode='reflect',   grid_mode=True)
        
 
    


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
        crs=CRS.from_user_input(4326),
        ):
    """downscale/disag the wse (s2) raster to match the dem resolution (s1)"""
    
    with Dsc_Session(crs=crs) as ses:
        log=ses.logger.getChild('m')
        #=======================================================================
        # precheck rasters
        #=======================================================================
        dem_stats, dem1_ar = rlay_extract(dem1_rlay_fp)
        wse_stats, wse2_ar = rlay_extract(wse2_rlay_fp)
        s2, s1 = dem_stats['res'][0], wse_stats['res'][0]
                    
        for stat in ['crs', 'bounds']:
            assert dem_stats[stat]==wse_stats[stat]
            
        assert dem_stats['crs']==ses.crs
        
        assert s2<s1, 'dem must have a finer resolution than the wse'
        
        if not s1%s2==0.0:
            log.warning(f'uneven resolution relation ({s1}/{s2}={s1%s2})')
            
        downscale=s1/s2
        
        log.info(f'downscaling from {s2} to {s1} ({downscale})')
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
        #cost distance the voids (some distance constraint?)
        #filter dem violators
        #smooth
        
        
        """option 2... 1) identify hydraulic blocks; 2) apply 1D weighted smoothing"""
        
        
        
        
        
        
        
            
        
 
        
        
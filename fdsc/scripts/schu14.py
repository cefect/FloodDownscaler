'''
Created on Jan. 25, 2023

@author: cefect

Scripts to replicate Schumann 2014's downscaling
'''


import os, datetime, shutil
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
from rasterio import shutil as rshutil
import geopandas as gpd
from sklearn.neighbors import KDTree, BallTree



from hp.rio import (
     write_resample, assert_extent_equal, Resampling, assert_spatial_equal,
     write_mask_apply, get_profile, write_array2, write_mosaic
     )

from hp.riom import (
    assert_mask_ar, load_mask_array, write_array_mask, write_extract_mask
    )

from hp.gpd import (
    raster_to_points, drop_z
    )

from fdsc.base import Dsc_basic, now


class Schuman14(Dsc_basic):
    
    def run_schu14(self, wse2_fp, dem_fp,
                   buffer_size=1.5,
                   gridcells=True,
 
                              **kwargs):
        """run python port of schuman 2014's downscaling
        
        Original script uses matlab's 'rangesearch' to match a 1.5 cell buffer
        
        
        Parameters
        ----------
        wse2_fp: str,
            filepath to coarse WSE (no phases)
            
        buffer_size: float, default 1.5
            size of buffer to include in downscale WSe search relative to the coarse resolution            
            see wbt.BufferRaster
            
        gridcells: bool, default False
            Optional flag to indicate that the 'size' threshold should be measured in grid cells instead of the default map units
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('schu14', subdir=False, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        assert_extent_equal(wse2_fp, dem_fp) 

        start = now()
        
        # get the downscale
        downscale = self.get_downscale(wse2_fp, dem_fp)
        
        meta_lib = {'smry':{
            'downscale':downscale, 'wse2_fp':os.path.basename(wse2_fp), 'dem_fp':dem_fp, 'ofp':ofp}}
        
        log.info(f'downscaling \'{os.path.basename(wse2_fp)}\' by {downscale} with buffer of {buffer_size}')
        #=======================================================================
        # get simple downscalled inundation
        #=======================================================================
        """ we want to allow buffer sizes as a fraction of the high-res grid
        """
        wse1_fp = write_resample(wse2_fp, scale=downscale, resampling=Resampling.nearest, out_dir=tmp_dir)
        assert_spatial_equal(dem_fp, wse1_fp)
        
        #=======================================================================
        # identify the search region
        #=======================================================================        
        search_mask_fp, meta_lib['searchzone'] = self.get_searchzone(wse1_fp, wbt_kwargs=dict(
            size=buffer_size * downscale, gridcells=gridcells), **skwargs)

        #=======================================================================
        # get the DEM within the search zone
        #=======================================================================
        srch_ar = load_mask_array(search_mask_fp, maskType='binary')
        demF_fp = write_mask_apply(dem_fp, srch_ar, logic=np.logical_or, ofp=os.path.join(tmp_dir, 'dem_search_masked.tif'))
        log.info(f'masked DEM to inundation search zone\n    {demF_fp}')
        
        #=======================================================================
        # populate valid WSE within the search zone (on the DEM)
        #=======================================================================
        wse1_filld_fp, meta_lib['knnF'] = self.get_knnFill(wse2_fp, demF_fp, **skwargs)
        
        #=======================================================================
        # merge
        #=======================================================================
        wse1_merge_fp = write_mosaic(wse1_fp, wse1_filld_fp, ofp=os.path.join(tmp_dir, 'wse_mosaic.tif'))
        
        #=======================================================================
        # wrap
        #======================================================================= 
        rshutil.copy(wse1_merge_fp, ofp)
        tdelta = (now() - start).total_seconds()
        meta_lib['smry']['tdelta'] = tdelta
        log.info(f'finished in {tdelta:.2f} secs')
        
        return ofp, meta_lib
        
    def get_knnFill(self, wse2_fp, dem_fp,   **kwargs):
        """
        fill the DEM with NN from the WSE (if higher)
        
        Parmaeters
        ----------
        wse2_fp: str
            filepath to coarse wse raster
 
            
        dem_fp: str
            filepath to fine DEm raster. should be masked so only zones-to-be-filled are un-masked
            
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('knnFill', subdir=False,  **kwargs)
        start = now()
        
        assert_extent_equal(wse2_fp, dem_fp)
 
        profile = get_profile(dem_fp)
        #=======================================================================
        # extract poinst
        #=======================================================================
        wse2_gser = raster_to_points(wse2_fp)
        
        #DEM points to populate
        dem_raw_gser = raster_to_points(dem_fp, drop_mask=False)
        bx = dem_raw_gser.geometry.z==-9999 #mask
        dem_gser =   dem_raw_gser[~bx]  
        
        #setup results frame
        res_gdf = gpd.GeoDataFrame(dem_gser.geometry.z.rename('dem'), geometry=drop_z(dem_gser.geometry))
        
        log.info(f'seraching from {len(dem_gser)} fine to {len(wse2_gser)} coarse')
        
        """
        ax = wse2_gser.plot(color='black')
        dem_gser.plot(color='green', ax=ax)
        
        res_gdf.plot(ax=ax, linewidth=0, column='wse2_id')
        """
        
        #=======================================================================
        # NN search
        #=======================================================================
        #convert point format
        src_pts = [(x,y) for x,y in zip(wse2_gser.geometry.x , wse2_gser.geometry.y)]
        qry_pts =  [(x,y) for x,y in zip(dem_gser.geometry.x , dem_gser.geometry.y)]
        
        #build model
        tree = KDTree(src_pts, leaf_size=3, metric='manhattan')
        
        #compute the distance and index to the source points (for each query point
        dist_ar, index_ar = tree.query(qry_pts, k=1, return_distance=True)        
        assert len(dist_ar)==len(qry_pts)
        
        #=======================================================================
        # join matches
        #=======================================================================
        res_gdf['wse2_id'] = pd.Series(index_ar.ravel(), index=dem_gser.index)
        
        res_gdf = res_gdf.join(wse2_gser.z.rename('wse2'), on='wse2_id').set_geometry('geometry', drop=True)
        
        assert res_gdf['wse2'].notna().all()
        
        res_gdf['wet'] = res_gdf['wse2']>res_gdf['dem'] #flag real water
        
        log.info(f'found %i/%i wet cells'%(res_gdf['wet'].sum(), len(res_gdf)))
        
        #=======================================================================
        # rasterize result
        #=======================================================================
        #add the empties back
        res_gdf2 = gpd.GeoDataFrame(res_gdf, index=dem_raw_gser.index, geometry=drop_z(dem_raw_gser.geometry)
                                    ).set_geometry('geometry', drop=True)
                                    
        res_gdf2['wet'] = res_gdf2['wet'].fillna(False).astype(bool)
        
        
        #recombine and resahep
        res_ar = ma.array(res_gdf2['wse2'].fillna(profile['nodata']).values, mask=~res_gdf2['wet'], fill_value=profile['nodata']
                          ).reshape((profile['height'], profile['width']))
                          
        write_array2(res_ar, ofp, **profile)
        
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now()-start).total_seconds()
        log.info(f'finished fill search in {tdelta} w/ %i/%i \n    {ofp}'%(
            res_gdf2['wet'].sum(), len(res_gdf2['wet'])))
        
        return ofp, dict(tdelta=tdelta, ofp=ofp, wet_cnt=res_gdf2['wet'].sum(), qry_cnt=len(qry_pts), src_cnt=len(src_pts))

 
        
        
        
        
 
        
    def get_searchzone(self, wse1_fp,
                   wbt_kwargs=dict(), **kwargs):
        """get a mask for the buffer search region"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('srch', subdir=False,  **kwargs)
 
        start = now()
        
        log.info('computing low-res buffer w/ %s'%wbt_kwargs)
        
 
        #=======================================================================
        # #convert inundation to mask
        #=======================================================================
        mask1_fp  = write_extract_mask(wse1_fp, out_dir=tmp_dir, maskType='binary')
        
        assert_spatial_equal(wse1_fp, mask1_fp)
        #=======================================================================
        # #make the buffer
        #=======================================================================
        buff1_fp = os.path.join(tmp_dir, 'buff1.tif')
        """this requires a binary type mask (just 1s and 0s"""
        assert self.buffer_raster(mask1_fp, buff1_fp, **wbt_kwargs)==0
        
        assert_spatial_equal(buff1_fp, wse1_fp)
        
        
        #=======================================================================
        # convert to donut
        #=======================================================================
        log.debug(f'computing donut on {buff1_fp}')
        #load the raw buffer
        buff1_ar = load_mask_array(buff1_fp, maskType='binary') 
        
        #load the original mask
        mask1_ar = load_mask_array(mask1_fp, maskType='binary')
 
        #inside buffer but outside wse
        
        new_mask = np.logical_and(np.invert(buff1_ar), mask1_ar)
        
        #write the new mask
        write_array_mask(np.invert(new_mask), ofp=ofp, maskType='binary', **get_profile(wse1_fp))
        assert_spatial_equal(ofp, wse1_fp)
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now()-start).total_seconds()
        log.info(f'built buffer in {tdelta} w/ {new_mask.sum()}/{new_mask.size} active \n    {ofp}')
        
        return ofp, dict(tdelta=tdelta, ofp=ofp, mask_fp=mask1_fp, valid_cnt=new_mask.sum())
    
    
    
    
    
def get_nearest(src_points, candidates, k_neighbors=2):
    """
    Find nearest neighbors for all source points from a set of candidate points
    modified from: https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
    and: https://stackoverflow.com/questions/62198199/k-nearest-points-from-two-dataframes-with-geopandas?noredirect=1&lq=1
    """
    

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='manhattan')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]
    closest_second = indices[1] # *manually add per comment above*
    closest_second_dist = distances[1] # *manually add per comment above*

    # Return indices and distances
    return (closest, closest_dist, closest_second, closest_second_dist)
    
    
    
    
        
        

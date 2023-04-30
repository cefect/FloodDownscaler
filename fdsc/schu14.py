'''
Created on Jan. 25, 2023

@author: cefect

Scripts to replicate Schumann 2014's downscaling
    see (https://github.com/cefect/Schu14_dscale) for original MatLab source code
'''


import os, datetime, shutil
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
from rasterio import shutil as rshutil
import geopandas as gpd
from sklearn.neighbors import KDTree, BallTree
from joblib import parallel_backend

from hp.basic import now
from hp.rio import (
     write_resample, assert_extent_equal, Resampling, assert_spatial_equal,
     write_mask_apply, get_profile, write_array2, write_mosaic
     )

from hp.riom import (
    assert_mask_ar, load_mask_array, write_array_mask, write_extract_mask
    )

from hp.gpd import (
    drop_z, set_mask, view
    )

from fdsc.base import DscBaseWorker, assert_partial_wet

methodName = 'Schumann14'

class Schuman14(DscBaseWorker):
    buffer_size=None
    
    def __init__(self, 
                 buffer_size=1.5,
                 n_jobs=6,
                 r2p_backend='gr',
                 run_dsc_handle_d=None,
                 **kwargs):
        """
        Parameters
        -----------
        n_jobs: int, default 6
            number of workers (threads or processes) that are spawned in parallel
            for sklearn
        """
        
        if run_dsc_handle_d is None: run_dsc_handle_d=dict()
        self.buffer_size=buffer_size
        self.n_jobs=n_jobs
        self.r2p_backend=r2p_backend
        
        if r2p_backend=='gr':
            assert_georasters()
            
        run_dsc_handle_d[methodName] = self.run_schu14 
        
        super().__init__(run_dsc_handle_d=run_dsc_handle_d, **kwargs)
    
    def run_schu14(self, wse_fp=None, dem_fp=None,
                   resampling=Resampling.nearest,
                   buffer_size=None,
                   gridcells=True,
 
                   r2p_backend=None,
                   downscale=None,
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
        start = now()        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('schu14', subdir=False, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        assert_extent_equal(wse_fp, dem_fp)
        
        
        
        if buffer_size is None: buffer_size=self.buffer_size
        if downscale is None:
            downscale = self.downscale
            
        if downscale is None:
            downscale = self.get_downscale(wse_fp, dem_fp, **skwargs)
        
        meta_lib = {'smry':{
            'downscale':downscale, 'wse2_fp':os.path.basename(wse_fp), 'dem_fp':dem_fp, 'ofp':ofp}}
        
        log.info(f'downscaling \'{os.path.basename(wse_fp)}\' by {downscale} with buffer of {buffer_size:.3f}')
        #=======================================================================
        # get simple downscalled inundation
        #=======================================================================
        """ we want to allow buffer sizes as a fraction of the high-res grid
        
        but we dont want to filter wet partials yet (no use in including thesein the search)
        
        knn search goes from the coarse WSE, so the resample method doesn't matter much here
        """
        wse1_rsmp_fp = write_resample(wse_fp, resampling=resampling,
                       scale=downscale,
                       ofp=self._get_ofp(dkey='wse1_resamp', out_dir=tmp_dir, ext='.tif'),
                       )
        
        meta_lib['smry']['wse1_rsmp_fp'] = wse1_rsmp_fp
        
        log.info(f'basic nearest resampled WSE {downscale} \n    {wse1_rsmp_fp}')       
        
        #=======================================================================
        # identify the search region
        #=======================================================================        
        search_mask_fp, meta_lib['searchzone'] = self.get_searchzone(wse1_rsmp_fp, wbt_kwargs=dict(
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
        wse1_filld_fp, meta_lib['knnF'] = self.get_knnFill(wse_fp, demF_fp,r2p_backend=r2p_backend, **skwargs)
        
        #=======================================================================
        # merge
        #=======================================================================
        wse1_merge_fp = write_mosaic(wse1_rsmp_fp, wse1_filld_fp, ofp=os.path.join(tmp_dir, 'wse1_mosaic.tif'))
        
        #=======================================================================
        # final filter
        #=======================================================================
        """should be for just the wet partials"""
        wse1_filter_ofp, meta_lib['WPfilter'] = self.get_wse_dem_filter(wse1_merge_fp, dem_fp,**skwargs)
        
        
        #=======================================================================
        # wrap
        #======================================================================= 
        rshutil.copy(wse1_filter_ofp, ofp)
        tdelta = (now() - start).total_seconds()
        meta_lib['smry']['tdelta'] = tdelta
        log.info(f'finished in {tdelta:.2f} secs on \n    {ofp}')
        
        return ofp, meta_lib
    
    def _get_gser(self, fp, backend=None,logger=None,**kwargs):
        """wrapper for pixels to point
        
        Parmaeters
        -----------
        backend: str,
            which backend to use for converting raster pixels to points
            tests show gr (GeoRaster) backend is fastest... but this requires an additional dependency
            
            
        TODO:
        add dependency checks
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = logger.getChild('pix2pts')
        start = now()
        if backend is None: backend=self.r2p_backend
        
        log.info(f'w/ \'{backend}\'')
        profile = get_profile(fp)
        #=======================================================================
        # load backend
        #=======================================================================
        if backend=='rio':
            from hp.rio_to_points import raster_to_points_windowed
            
            func = lambda x:raster_to_points_windowed(x, drop_mask=True, max_workers=os.cpu_count())
        elif backend=='rio_simple':
            from hp.rio_to_points import raster_to_points_simple
            func = lambda x:raster_to_points_simple(x, drop_mask=True, max_workers=1)
        elif backend=='gr':
            assert_georasters()
            from hp.gr import pixels_to_points as func
        else:
            raise KeyError(backend)
            
        #=======================================================================
        # execute
        #=======================================================================
        gser = func(fp, **kwargs)
        
        #=======================================================================
        # filter
        #=======================================================================
        """switch to always drop to be consistent with GeoRaster
        #check shape
        rsize = profile['width']*profile['height']
        assert len(gser)==rsize"""
        
        assert not np.any(gser.z==-9999)
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now() - start).total_seconds()
        
        log.info(f'finished in {tdelta} with {len(gser)}')
        
        return gser
            
 
            
            
            
        
        
    def get_knnFill(self, wse2_fp, dem_fp, n_jobs=None, 
                    r2p_backend=None, 
 
                    **kwargs):
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
        skwargs=dict(logger=log, backend=r2p_backend)
        start = now()
        if n_jobs is None: n_jobs=self.n_jobs
        
        assert_extent_equal(wse2_fp, dem_fp)
 
        profile = get_profile(dem_fp)
        assert profile['nodata']==-9999
        #=======================================================================
        # extract poinst
        #=======================================================================
        log.debug(f'converting raster_to_points on {wse2_fp}')
        wse2_gser = self._get_gser(wse2_fp, **skwargs)
        
        #DEM points to populate
        log.debug(f'converting raster_to_points on {dem_fp}')
        dem_gser = self._get_gser(dem_fp,**skwargs)
        
 
        
        #setup results frame
        log.debug('building results frame')
        res_gdf = gpd.GeoDataFrame(dem_gser.geometry.z.rename('dem'), geometry=drop_z(dem_gser.geometry))
        
        #=======================================================================
        # NN search
        #=======================================================================
        log.info(f'seraching from {len(dem_gser)} fine to {len(wse2_gser)} coarse')
        
        """
        ax = wse2_gser.plot(color='black')
        dem_gser.plot(color='green', ax=ax)
        
        res_gdf.plot(ax=ax, linewidth=0, column='wse2_id')
        
        view(res_gdf)
        """
        
        #convert point format
        src_pts = [(x,y) for x,y in zip(wse2_gser.geometry.x , wse2_gser.geometry.y)]
        qry_pts =  [(x,y) for x,y in zip(dem_gser.geometry.x , dem_gser.geometry.y)]
        
        #build model
        with parallel_backend('threading', n_jobs=n_jobs): 
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
 
        #dump points to file
        pts_fp = os.path.join(tmp_dir, 'wse1_match_wet_pts.shp')
        res_wet_gdf = res_gdf.loc[res_gdf['wet'], :].reset_index()
 
        res_wet_gdf.loc[:, ['wse2']].set_geometry(res_wet_gdf.geometry).to_file(pts_fp, layer='match_pts', driver="ESRI Shapefile")
        
        #wbt rasterize
        #wse1_new_rlay_fp = os.path.join(tmp_dir, 'wse1_match_rasterized.tif')
        assert self.vector_points_to_raster(pts_fp, ofp, field='wse2', nodata=profile['nodata'], base=dem_fp)==0
        
 
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now()-start).total_seconds()
        log.info(f'finished KNN searchc + fill in {tdelta} secs w/ %i/%i \n    {ofp}'%(
            res_gdf['wet'].sum(), len(res_gdf['wet'])))
        
        return ofp, dict(tdelta=tdelta, ofp=ofp, wet_cnt=res_gdf['wet'].sum(), qry_cnt=len(qry_pts), src_cnt=len(src_pts))
        
    def get_searchzone(self, wse1_fp,
                   wbt_kwargs=dict(), **kwargs):
        """get a mask for the buffer search region"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('srch', subdir=False, **kwargs)
 
        start = now()
        
        log.info('computing low-res buffer w/ %s' % wbt_kwargs)
 
        #=======================================================================
        # #convert inundation to mask
        #=======================================================================
        mask1_fp = write_extract_mask(wse1_fp, out_dir=tmp_dir, maskType='binary')
        
        assert_spatial_equal(wse1_fp, mask1_fp)
        #=======================================================================
        # #make the buffer
        #=======================================================================
        buff1_fp = os.path.join(tmp_dir, 'buff1.tif')
        """this requires a binary type mask (just 1s and 0s"""
        assert self.buffer_raster(mask1_fp, buff1_fp, **wbt_kwargs) == 0
        
        assert_spatial_equal(buff1_fp, wse1_fp)
        
        #=======================================================================
        # convert to donut
        #=======================================================================
        log.debug(f'computing donut on {buff1_fp}')
        # load the raw buffer
        buff1_ar = load_mask_array(buff1_fp, maskType='binary') 
        
        # load the original mask
        mask1_ar = load_mask_array(mask1_fp, maskType='binary')
 
        # inside buffer but outside wse        
        new_mask = np.logical_and(np.invert(buff1_ar), mask1_ar)
        assert_partial_wet(new_mask, msg='search donut')
        
        # write the new mask
        write_array_mask(np.invert(new_mask), ofp=ofp, maskType='binary', **get_profile(wse1_fp))
        assert_spatial_equal(ofp, wse1_fp)
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now() - start).total_seconds()
        log.info(f'built search zone buffer in {tdelta} w/ {new_mask.sum()}/{new_mask.size} active \n    {ofp}')
        
        return ofp, dict(tdelta=tdelta, ofp=ofp, mask_fp=mask1_fp, valid_cnt=new_mask.sum())
    
    
#===============================================================================
# def get_nearest(src_points, candidates, k_neighbors=2):
#     """
#     Find nearest neighbors for all source points from a set of candidate points
#     modified from: https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html
#     and: https://stackoverflow.com/questions/62198199/k-nearest-points-from-two-dataframes-with-geopandas?noredirect=1&lq=1
#     """
#     
# 
#     # Create tree from the candidate points
#     tree = BallTree(candidates, leaf_size=15, metric='manhattan')
# 
#     # Find closest points and distances
#     distances, indices = tree.query(src_points, k=k_neighbors)
# 
#     # Transpose to get distances and indices into arrays
#     distances = distances.transpose()
#     indices = indices.transpose()
# 
#     # Get closest indices and distances (i.e. array at index 0)
#     # note: for the second closest points, you would take index 1, etc.
#     closest = indices[0]
#     closest_dist = distances[0]
#     closest_second = indices[1] # *manually add per comment above*
#     closest_second_dist = distances[1] # *manually add per comment above*
# 
#     # Return indices and distances
#     return (closest, closest_dist, closest_second, closest_second_dist)
#===============================================================================
    
    
    
import pkg_resources
def assert_georasters(v='0.5.24'):
    """check that the dependency is installed"""
    if not __debug__: # true if Python was not started with an -O option
        return
 
    __tracebackhide__ = True
    
    try:
        package = pkg_resources.get_distribution('georasters')
        version = package.version
        assert pkg_resources.parse_version(version) >= pkg_resources.parse_version(v), \
               "georasters version is below " + v
    except pkg_resources.DistributionNotFound:
        raise AssertionError("dependency \'georasters\' not found")

        
        

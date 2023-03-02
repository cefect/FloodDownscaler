'''
Created on Jan. 6, 2023

@author: cefect

validating downscaling results
'''
import logging, os, copy, datetime, pickle
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
import rasterio.features 
import geopandas as gpd
from sklearn.metrics import confusion_matrix

from hp.rio import (
    RioSession, RioWrkr, assert_rlay_simple, get_stats, assert_spatial_equal, get_depth,is_raster_file,
    write_array2,get_write_kwargs
    )

from hp.riom import write_array_mask

from hp.gpd import (
    get_samples, GeoPandasWrkr,
    )
from hp.logr import get_new_console_logger
from hp.err_calc import ErrorCalcs, get_confusion_cat

from fdsc.base import (
    Master_Session, assert_partial_wet, rlay_extract, assert_wse_ar
    )
# from definitions import src_name


class ValidateGrid(RioWrkr):
    """compute validation metrics for a raster by comparing to some true raster""" 
    
    confusion_ser = None
    confusion_codes = {'TP':111, 'TN':110, 'FP':101, 'FN':100}
    
    pred_fp = None
    true_fp = None
    
    pred_ar=None
    pred_stats_d=None
    
    def __init__(self,
                 true_fp=None,
                 pred_fp=None,
 
                 logger=None,
                 **kwargs):
        
        #=======================================================================
        # pre init
        #=======================================================================
        if logger is None:
            logger = get_new_console_logger(level=logging.DEBUG)
        
        super().__init__(logger=logger, **kwargs)
                
        #=======================================================================
        # load rasters
        #=======================================================================
        """using ocnditional loading mostly for testing"""
        
        if not pred_fp is None: 
            self._load_pred(pred_fp)
        
        if not true_fp is None:
            if not is_raster_file(true_fp):
                rlay_fp = self._rasterize_inundation(true_fp)
            else:
                rlay_fp=true_fp
                
            self._load_true(rlay_fp) 
            

            
    def _load_true(self, rlay_fp):

        #=======================================================================
        # rasterize polygon
        #=======================================================================

            
            
        stats_d, self.true_ar = rlay_extract(rlay_fp)
        assert_wse_ar(self.true_ar, msg='true array')
        self.logger.info('loaded true raster from file w/\n    %s' % stats_d)
        # set the session defaults from this
        if 'dtypes' in stats_d:
            stats_d['dtype'] = stats_d['dtypes'][0]
        self.stats_d = stats_d
        self._set_defaults(stats_d)
        self.true_fp = rlay_fp
        
        if not self.pred_fp is None:
            assert_spatial_equal(self.true_fp, self.pred_fp)

    def _load_pred(self, pred_fp):
        self.pred_stats_d, self.pred_ar = rlay_extract(pred_fp)
        assert_wse_ar(self.pred_ar, msg='pred array')
        
        self.pred_fp = pred_fp
        
        if not self.true_fp is None:
            assert_spatial_equal(self.true_fp, self.pred_fp)
            
        self.logger.info('loaded pred raster from file w/\n    %s' % self.pred_stats_d)
        
    def _rasterize_inundation(self, poly_fp,
                              out_shape=None,
                              transform=None,
                              dtype=None,nodata=None,
                               **kwargs):
        """convert polygonized inundation into a raster"""
        
        raise IOError('make this a standalone function that takes a reference raster')
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, _, resname = self._func_setup('rasterize', **kwargs)
        ext = os.path.splitext(poly_fp)[1]
        ofp = os.path.join(out_dir, os.path.basename(poly_fp).replace(ext, '.tif'))
        
        #copy spatial defaults from the predicted array
        if out_shape is None:
            out_shape=self.pred_ar.shape
        
        if transform is None:
            transform=self.pred_stats_d['transform']
            
        if dtype is None:
            dtype = self.pred_ar.dtype
            
        if nodata is None:
            nodata=self.pred_stats_d['nodata']
            
 
 
        log.info(f'building raster from {poly_fp}')
        
        
        assert not self.pred_stats_d is None, 'need to load a predicted array first'
        gdf = gpd.read_file(poly_fp)
        assert len(gdf)==1
        
        #get an array from this
        ar = rasterio.features.rasterize(gdf.geometry,all_touched=False,
                                         fill=nodata,
                                         out_shape=out_shape,transform=transform,  dtype=dtype,
                                         )
        mar = ma.array(ar, mask=ar==nodata, fill_value=nodata)
        #write to raster
        return write_array2(mar, ofp,  **get_write_kwargs(self.pred_fp))
            
 
    
    #===========================================================================
    # grid inundation metrics------
    #===========================================================================
    def get_hitRate(self, **kwargs):
        """proportion of wet benchmark data that was replicated by the model"""
        # log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP'] / (cf_ser['TP'] + cf_ser['FN'])
    
    def get_falseAlarms(self, **kwargs):
        # log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['FP'] / (cf_ser['TP'] + cf_ser['FP'])
    
    def get_criticalSuccessIndex(self, **kwargs):
        """critical success index. accounts for both overprediction and underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP'] / (cf_ser['TP'] + cf_ser['FP'] + cf_ser['FN'])
    
    def get_errorBias(self, **kwargs):
        """indicates whether the model has a tendency toward overprediction or underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        return cf_ser['FP'] / cf_ser['FN']
    
    def get_inundation_all(self, **kwargs):
        """convenience for getting all the inundation metrics
        NOT using notation from Wing (2017)
        """
        
        d = {
            'hitRate':self.get_hitRate(**kwargs),
            'falseAlarms':self.get_falseAlarms(**kwargs),
            'criticalSuccessIndex':self.get_criticalSuccessIndex(**kwargs),
            'errorBias':self.get_errorBias(**kwargs),
            }
        
        # add confusion codes
        d.update(self.confusion_ser.to_dict())
        
        #=======================================================================
        # #checks
        #=======================================================================
        assert set(d.keys()).symmetric_difference(
            ['hitRate', 'falseAlarms', 'criticalSuccessIndex', 'errorBias', 'TN', 'FP', 'FN', 'TP']
            ) == set()
            
        assert pd.Series(d).notna().all(), d
        
        self.logger.info('computed all inundation metrics:\n    %s' % d)
        return d

    def get_confusion_grid(self,
                           confusion_codes=None, **kwargs):
        """generate confusion grid
        
        Parameters
        ----------
        confusion_codes: dict
            integer codes for confusion labels
        """
        log, true_ar, pred_ar = self._func_setup_local('confuGrid', **kwargs)
        
        if confusion_codes is None: confusion_codes = self.confusion_codes
        
        # convert to boolean (true=wet=nonnull)
        true_arB, pred_arB = np.invert(true_ar.mask), np.invert(pred_ar.mask)
        
        res_ar = get_confusion_cat(true_arB, pred_arB, confusion_codes=confusion_codes)
        
        #=======================================================================
        # check
        #=======================================================================
        if __debug__:
            cf_ser = self._confusion(true_ar=true_ar, pred_ar=pred_ar, **kwargs)
            
            # build a frame with the codes
            df1 = pd.Series(res_ar.ravel(), name='grid_counts').value_counts().to_frame().reset_index()            
            df1['index'] = df1['index'].astype(int) 
            df2 = df1.join(pd.Series({v:k for k, v in confusion_codes.items()}, name='codes'), on='index'
                           ).set_index('index')
                           
            # join the values from sklearn calcs
            df3 = df2.join(cf_ser.rename('sklearn_counts').reset_index().rename(columns={'index':'codes'}).set_index('codes'),
                     on='codes')
            
            compare_bx = df3['grid_counts'] == df3['sklearn_counts']
            if not compare_bx.all():
                raise AssertionError('confusion count mismatch\n    %s' % compare_bx.to_dict())
            
        log.info('finished on %s' % str(res_ar.shape))
        
        return res_ar
 
    #===========================================================================
    # private helpers------
    #===========================================================================
    def _confusion(self, **kwargs):
        """retrieve or construct the wet/dry confusion series"""
        if self.confusion_ser is None:
            log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
            
            # convert to boolean (true=wet=nonnull)
            true_arB, pred_arB = np.invert(true_ar.mask), np.invert(pred_ar.mask)
            
            assert_partial_wet(true_arB)
            assert_partial_wet(pred_arB)
  
            # fancy labelling
            self.confusion_ser = pd.Series(confusion_matrix(true_arB.ravel(), pred_arB.ravel(),
                                                            labels=[False, True]).ravel(),
                      index=['TN', 'FP', 'FN', 'TP'])
            
            assert self.confusion_ser.notna().all(), self.confusion_ser
            
            log.info('generated wet-dry confusion matrix on %s\n    %s' % (
                str(true_ar.shape), self.confusion_ser.to_dict()))
            
        return self.confusion_ser.copy()

    def _func_setup_local(self, dkey,
                    logger=None,
                    true_ar=None, pred_ar=None,
                    ):
        """common function default setup"""
 
        if logger is None:
            logger = self.logger
        log = logger.getChild(dkey)
        
        if true_ar is None: true_ar = self.true_ar
        if pred_ar is None: pred_ar = self.pred_ar
            
        return log, true_ar, pred_ar
    

class ValidatePoints(ValidateGrid, GeoPandasWrkr):
    """methods for validation with points"""
    
    stats_d = None
    pts_gdf = None
    sample_pts_fp = None
    dem_fp = None
    
    def __init__(self,
 
                 sample_pts_fp=None,
                 dem_fp=None,
                 index_coln='id',

                 **kwargs):
        
        #=======================================================================
        # pre init
        #=======================================================================
        
        super().__init__(**kwargs)
                
        #=======================================================================
        # load poitns
        #=======================================================================
        self.index_coln = index_coln
        if not dem_fp is None:
            self.dem_fp = dem_fp
        if not sample_pts_fp is None:
            self._load_pts(sample_pts_fp, index_coln=index_coln)
            
    def _load_stats(self, fp=None):
        """set session stats from a raster
        
        mostly used by tests where we dont load the raster during init"""
        
        if fp is None:
            fp = self.dem_fp
            
        assert not fp is None
        
        with rio.open(fp, mode='r') as ds:
            assert_rlay_simple(ds)
            self.stats_d = get_stats(ds) 
 
    def _load_pts(self, fp, index_coln=None, bbox=None):
        """load sample points"""
        if index_coln is None: index_coln = self.index_coln
        assert os.path.exists(fp)
        
        # load raster stats
        if self.stats_d is None:
            self._load_stats()
        
        # get bounding box from rasters
        if bbox is None:
            bbox = self.stats_d['bounds']
        
        gdf = gpd.read_file(fp, bbox=bbox)
        
        # check
        assert gdf.crs == self.stats_d['crs']
        assert (gdf.geometry.geom_type == 'Point').all()
        
        # clean
        self.pts_gser = gdf.set_index(index_coln).geometry
        self.sample_pts_fp = fp
        
    def get_samples(self,
                           true_fp=None,
                           pred_fp=None,
                           sample_pts_fp=None,
                           gser=None,
                           **kwargs):
        """sample raster with poitns"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('samps', **kwargs)
        
        if true_fp is None:
            true_fp = self.true_fp
        if pred_fp is None:
            pred_fp = self.pred_fp
            
        if not sample_pts_fp is None:
            assert gser is None
            self._load_stats(true_fp)
            self._load_pts(sample_pts_fp)
            
        if gser is None:
            gser = self.pts_gser
 
        #=======================================================================
        # sample each
        #=======================================================================
        log.info(f'sampling {len(gser)} pts on 2 rasters')
        d = dict()
        for k, fp in {'true':true_fp, 'pred':pred_fp}.items():
            log.info(f'sampling {k}')
            with rio.open(fp, mode='r') as ds:
                d[k] = get_samples(gser, ds, colName=k).drop('geometry', axis=1)
        
        samp_gdf = pd.concat(d.values(), axis=1).set_geometry(gser)
        
        #=======================================================================
        # wrap
        #=======================================================================
        assert not samp_gdf.isna().any().any(), 'no nulls.. should be depths'
        
        log.info(f'finished sampling w/ {str(samp_gdf.shape)}')
        
        return samp_gdf
    
    def get_samp_errs(self, gdf_raw, **kwargs):
        """calc errors between pred and true"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('samp_errs', **kwargs)
        
        #=======================================================================
        # clean
        #=======================================================================
        gdf = gdf_raw.drop('geometry', axis=1)  # .dropna(how='any', subset=['true'])
        
        assert gdf.notna().all().all()
        
        #=======================================================================
        # calc
        #=======================================================================
        
        with ErrorCalcs(pred_ser=gdf['pred'], true_ser=gdf['true'], logger=log) as wrkr:
            err_d = wrkr.get_all(dkeys_l=['bias', 'meanError', 'meanErrorAbs', 'RMSE', 'pearson'])
            
            # get confusion
            _, cm_dx = wrkr.get_confusion(wetdry=True, normed=False)            
            err_d.update(cm_dx.droplevel([1, 2])['counts'].to_dict())
            
        return err_d

    
class ValidateSession(ValidatePoints, RioSession, Master_Session):

    def __init__(self,
                 run_name=None,
                 **kwargs):
 
        if run_name is None:
            run_name = 'vali_v1'
        super().__init__(run_name=run_name, **kwargs)

    def run_vali_pts(self, sample_pts_fp,
                            true_fp=None,
                           pred_fp=None,
                           **kwargs):
        """validation on poitns pipeline
        assumes rasters are depth rasters"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('pts', subdir=False, ext='.gpkg', **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir)
 
        #=======================================================================
        # #sample points
        #=======================================================================
        gdf = self.get_samples(true_fp=true_fp, pred_fp=pred_fp, sample_pts_fp=sample_pts_fp, **skwargs)
        
        # write
        meta_d = {'sample_pts_fp':sample_pts_fp, 'cnt':len(gdf)}
        # ofpi = self._get_ofp(out_dir=out_dir, dkey='samples', ext='.geojson')
        gdf.to_file(ofp, crs=self.crs)
        meta_d['samples_fp'] = ofp
        log.info(f'wrote {len(gdf)} to \n    {ofp}')
        #=======================================================================
        # #calc errors
        #=======================================================================
        err_d = self.get_samp_errs(gdf, **skwargs)
        # meta
        meta_d.update(err_d)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        return err_d, meta_d

    def run_vali(self,
                 true_fp=None, pred_fp=None,
                 sample_pts_fp=None, dem_fp=None,
                 write_meta=True,
                 **kwargs):
        """
        run all validations on a downsampled grid (compared to a true grid).
            called by pipeline.run_dsc_vali()
        
        
        Parameters
        -----------
        
        sample_pts_fp: str, optional
            filepath to points vector layer for sample-based metrics
            
        dem_fp: str, optional
            filepath to dem (for converting to depths)
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('vali', subdir=True, **kwargs)
        meta_lib = {'smry':{**{'today':self.today_str}, **self._get_init_pars()}}
        metric_lib = dict()
        skwargs = dict(logger=log)
        #=======================================================================
        # load
        #=======================================================================
        if not true_fp is None:
            self._load_true(true_fp) 
        else:
            true_fp = self.true_fp
            
        if not pred_fp is None: 
            self._load_pred(pred_fp)
        else:
            pred_fp = self.pred_fp
            
        if sample_pts_fp is None:
            sample_pts_fp = self.sample_pts_fp
            
        if dem_fp is None:
            dem_fp = self.dem_fp
        
        #=======================================================================
        # precheck
        #=======================================================================
        true_ar, pred_ar = self.true_ar, self.pred_ar
        assert isinstance(pred_ar, np.ndarray)
        
        # data difference check
        if (true_ar == pred_ar).all():
            raise IOError('passed identical grids')
        if (true_ar.mask == pred_ar.mask).all():
            log.warning('identical masks on pred and true')
        
        #=======================================================================
        # grid metrics
        #=======================================================================        
        shape, size = true_ar.shape, true_ar.size
        meta_lib['grid'] = {**{'shape':str(shape), 'size':size, 'true_fp':true_fp, 'pred_fp':pred_fp},
                            **copy.deepcopy(self.stats_d)}
        
        #=======================================================================
        # inundation metrics-------
        #=======================================================================
        log.info(f'computing inundation metrics on %s ({size})' % str(shape))
        
        # confusion_ser = self._confusion(**skwargs)
        inun_metrics_d = self.get_inundation_all(**skwargs)
        
        # confusion grid
        confusion_grid_ar = self.get_confusion_grid(**skwargs)
 
        # meta
        meta_d = copy.deepcopy(inun_metrics_d)
        
        #=======================================================================
        # write
        #=======================================================================
        meta_d['confuGrid_fp'] = self.write_array(confusion_grid_ar, out_dir=out_dir,
                                                       resname=self._get_resname(dkey='confuGrid'))
 
        meta_lib['inun'] = meta_d
        metric_lib['inun'] = inun_metrics_d
        
        #=======================================================================
        # asset samples---------
        #=======================================================================            
        if not sample_pts_fp is None:
            assert isinstance(dem_fp, str), type(dem_fp)
            # build depth grids
            true_dep_fp = get_depth(dem_fp, true_fp, ofp=self._get_ofp(out_dir=out_dir, resname='true_dep'))
            pred_dep_fp = get_depth(dem_fp, pred_fp, ofp=self._get_ofp(out_dir=tmp_dir, resname='pred_dep'))
            
            metric_lib['samp'], meta_lib['samp'] = self.run_vali_pts(sample_pts_fp,
                                        true_fp=true_dep_fp, pred_fp=pred_dep_fp, logger=log, out_dir=out_dir)
            
            meta_lib['grid']['true_dep_fp'] = true_dep_fp
            meta_lib['grid']['dep1'] = pred_dep_fp
        
        #=======================================================================
        # wrap-----
        #=======================================================================        
        if write_meta:
            self._write_meta(meta_lib, logger=log, out_dir=out_dir)
        
        log.info('finished')
        return metric_lib, meta_lib

    
def run_validator(true_fp, pred_fp,
        **kwargs):
    """compute error metrics and layers on a wse layer"""
    
    with ValidateSession(true_fp=true_fp, pred_fp=pred_fp, **kwargs) as ses:
        res = ses.run_vali()
        
    return res
    

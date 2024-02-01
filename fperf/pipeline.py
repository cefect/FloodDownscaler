'''
Created on Mar. 3, 2023

@author: cefect
'''
import logging, os, copy, datetime, pickle, math, shutil
import numpy as np
import shapely.geometry as sgeo
import pandas as pd
import rasterio as rio
import rasterio.shutil


from hp.basic import get_dict_str, dstr
from hp.rio import (
    RioSession,  assert_rlay_simple, _get_meta, assert_spatial_equal, is_raster_file,
    write_array2, rlay_ar_apply, get_meta, get_ds_attr, write_clip, assert_extent_equal,
    is_spatial_equal, get_support_ratio, get_profile, get_shape, is_raster_file, copyr
    )

import hp.riom
from hp.riom import rlay_mar_apply, write_extract_mask, assert_mask_fp

from hp.gpd import (
    write_rasterize, get_samples, assert_file_points
    )

from hp.hyd import (
    assert_partial_wet, assert_wse_ar, assert_wsh_ar, assert_dem_ar, write_wsh_boolean,
    get_wsh_rlay, HydTypes
    )

from hp.fiona import get_bbox_and_crs
 

#from fdsc.base import (Master_Session,   rlay_extract,)
from fperf.base import BaseSession
from fperf.inun import ValidateMask
from fperf.wd import ValidatePoints

class ValidateSession(ValidateMask, ValidatePoints, RioSession, BaseSession):

    def __init__(self,
                 run_name=None,
                 **kwargs):
 
        if run_name is None:
            run_name = 'pipe1'
        super().__init__(run_name=run_name, **kwargs)
        
        self.inun_rlay_fp_lib = dict()
        
    def build_random_wsh_analog(self,
                     ref_rlay_fp,
                     zero_frac=0.5,
                     rescale=None,
                     
                     **kwargs):
        """build a dummy random raster similar to a passed raster"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rand', ext='.tif',  **kwargs)
        
        #=======================================================================
        # load reference data
        #=======================================================================
        log.info(f'building random analog from {ref_rlay_fp}')
        with rio.open(ref_rlay_fp, mode='r') as ds:
            prof = ds.profile
            #shape = (prof['height'], prof['width'])
            raw_ar = ds.read(1, masked=True)
            
 
        #=======================================================================
        # build the rando
        #=======================================================================
        
        #from zero to 1
        ar1 = np.random.random(raw_ar.shape)
        
        #add zeros randomly
        """needed for WSH assertions"""
        if not zero_frac is None:
            log.info(f'forcing {zero_frac} to be zeros')
            c = int(math.ceil(ar1.size*zero_frac)) #indexes
            ar1.ravel()[np.random.choice(ar1.size, c, replace=False)] = 0.0
            assert (ar1==0.0).sum()/ar1.size==zero_frac
            
            
 
        
        """
        ar2.min()
        import matplotlib.pyplot as plt
        plt.show()
        """
        
        #scale
        if rescale is None:
            ser = pd.Series(raw_ar.data.ravel())
            rescale = (ser.quantile(.98) - ser.quantile(.02)) 
            
               
        ar_scaled = ar1*rescale
        
        #=======================================================================
        # write arrasy
        #=======================================================================
        log.info(f'rescaled {raw_ar.shape} by {rescale} and writing to \n    {ofp}') 
        return write_array2(ar_scaled, ofp, **prof)
        
 
        
    def run_vali_inun(self,
               true_inun_fp=None,
                 pred_inun_fp=None,
                 
                 **kwargs):
        
        """run inundation validation sequence
        
        Pars
        --------
        true_inun_fp: str
            filepath to inundation raster 
            default assumes WSE. (only the mask is loaded) wet=0
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('inun',  **kwargs)
 
        skwargs = dict(logger=log) #these funcs are using a specical local setup
 
        self._set_profile(rlay_ref_fp=pred_inun_fp)
        #=======================================================================
        # load
        #=======================================================================
        assert_spatial_equal(true_inun_fp, pred_inun_fp)
        
        """isolating w/ context management"""
        with ValidateMask(out_dir=out_dir, logger=log, tmp_dir=tmp_dir,
                          proj_name=self.proj_name, wrk_dir=self.wrk_dir,
                          ) as wrkr: 
        
            wrkr._load_mask_true(true_inun_fp, invert=False)  #sets worker defaults
            wrkr._load_mask_pred(pred_inun_fp, invert=False)
            
     
            fp_d = dict(true_inun_fp=true_inun_fp, pred_inun_fp=pred_inun_fp)
            mar = wrkr.true_mar
            #=======================================================================
            # precheck
            #=======================================================================
            wrkr._check_inun()
            
            #=======================================================================
            # inundation metrics-------
            #=======================================================================
            log.info(f'computing inundation metrics on {mar.shape} ({mar.size})')
            
            # confusion_ser = self._confusion(**skwargs)
            inun_metrics = wrkr.get_inundation_all(**skwargs)
            
            # confusion grid
            confusion_grid_ar = wrkr.get_confusion_grid(**skwargs)        
            fp_d['confuGrid_fp'] = self.write_array(confusion_grid_ar, 
                                                    out_dir=out_dir,
                                                    resname=self._get_resname(dkey='confuGrid'))
            
            
        
        #=======================================================================
        # wrap
        #=======================================================================
        meta_d = {'shape':str(mar.shape)}
        meta_d.update(inun_metrics)
        meta_d.update(fp_d)
        
        log.info(f'finished w/ \n    {inun_metrics}')
        
        return inun_metrics, fp_d, meta_d
    
    
    
    def run_vali_confuSamps(self,
                            confuGrid_fp, sample_pts_fp,
                            **kwargs):
        """sample confusion grid with points"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('confuS', subdir=True, ext='.geojson', **kwargs)
        
        log.info(f'computing inundation metrics on samples from {os.path.basename(sample_pts_fp)}')
        
        #load points
        gdf = self._get_gdf(sample_pts_fp, stats_d=get_meta(confuGrid_fp))
        
        
        # get values from raster 
        log.info(f'sampling {os.path.basename(confuGrid_fp)} on points')
        with rio.open(confuGrid_fp, mode='r') as ds:
            confu_gdf = get_samples(gdf.geometry, ds, colName='confusion')
            
        #=======================================================================
        # #write
        #=======================================================================
 
        confu_gdf.to_file(ofp, crs=self.crs)
 
        log.info(f'wrote {len(gdf)} to \n    {ofp}')
        
        return ofp
        
 
 

    def run_vali_pts(self, sample_pts_fp,
                           true_wd_fp=None,
                           pred_wd_fp=None,
                           **kwargs):
        """comprae two depth rasters at points"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('pts', subdir=True, ext='.geojson', **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir)
 
        #=======================================================================
        # #sample points
        #=======================================================================
        gdf = self.get_samples(true_wd_fp=true_wd_fp, pred_wd_fp=pred_wd_fp, sample_pts_fp=sample_pts_fp, **skwargs)
        
        # write
        meta_d = {'sample_pts_fp':sample_pts_fp, 'cnt':len(gdf)}
 
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
    
    def run_vali_hwm(self, wd_fp, hwm_fp, wd_key=None,
                     **kwargs):
        """compare a depth raster against some point values"""
        #=======================================================================
        # defautls
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('hwm', ext='.geojson', **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir)
        if wd_key is None: wd_key=self.wd_key
        fp_d = {'hwm_pts_fp':hwm_fp,'wd_fp':wd_fp}
        #=======================================================================
        # load points
        #=======================================================================
        rlay_stats_d = get_meta(wd_fp)
        
        gdf = self._get_gdf(hwm_fp, stats_d=rlay_stats_d)
        
        assert len(gdf)>0, f'failed to load any HWM points from \n    {hwm_fp}\n    AOI too small?'
        assert wd_key in gdf
        
        log.info(f'loaded {len(gdf)} HWMs from \n    {hwm_fp}') 
        
        #=======================================================================
        # get values from raster
        #=======================================================================
        log.info(f'sampling {os.path.basename(wd_fp)} on points')
        with rio.open(wd_fp, mode='r') as ds:
            gdf = gdf.join(
                get_samples(gdf.geometry, ds, colName='pred').drop('geometry', axis=1)
                ).drop('geometry', axis=1).set_geometry(gdf.geometry).rename(columns={wd_key:'true'})
                
        
        # write
        
        meta_d = {'cnt':len(gdf)}
 
        gdf.to_file(ofp, crs=self.crs)
        fp_d['pred_hwm_fp'] = ofp
        log.info(f'wrote {len(gdf)} to \n    {ofp}')
        
        #=======================================================================
        # #calc errors
        #=======================================================================
        err_d = self.get_samp_errs(gdf, **skwargs)
        
        # meta
        meta_d.update(err_d)
        meta_d.update(fp_d)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished')
        return err_d, fp_d, meta_d
    
 

    def run_vali(self,
                 pred_wse_fp=None, 
                 true_wse_fp=None,
                 true_inun_fp=None,
                 sample_pts_fp=None,
                 hwm_pts_fp=None,
                 dem_fp=None,
                 write_meta=True,
                 **kwargs):
        """
        run all validations on a downsampled grid (compared to a true grid).
            called by pipeline.run_dsc_vali()
            
            allows separate inundation and wse validation
                or just uses wse
                
            allows inundation to be a polygon
        
        
        Parameters
        -----------
        pred_wse_fp: str
            predicted WSE grid. used to build WD. 
            
        true_wse_fp: str
            valid WSE grid. used to build inun grid if its not passed
            
        true_inun_fp: str, optional
            valid inundation extents (rlay or vlay). uses true_wse_fp if not passed
        
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
        skwargs = dict(logger=log, out_dir=out_dir)
        fp_d = dict(dem_fp=dem_fp, pred_wse_fp=pred_wse_fp, true_wse_fp=true_wse_fp, 
                    true_inun_fp=true_inun_fp,  hwm_pts_fp=hwm_pts_fp) #for reporting
        
        log.info(f'running validation on \n    %s'%(
            {k:os.path.basename(v) for k,v in fp_d.items() if not v is None}))
        #=======================================================================
        # common prep
        #=======================================================================
        #dem
        assert isinstance(dem_fp, str), type(dem_fp)
        rlay_ar_apply(dem_fp, assert_dem_ar)        
 
        #pred
        if pred_wse_fp is None:
            raise NotImplementedError('need to passe a wse')
    
        meta_lib['grid'] = get_meta(pred_wse_fp)
        
        rlay_ar_apply(pred_wse_fp, assert_wse_ar)
        
        assert_spatial_equal(dem_fp, pred_wse_fp)
        
        #helper func
        def clip_rlay(rlay_fp):
            """clip the raster to the predicted bounds if necessary"""
            pd = meta_lib['grid']
            ibounds = get_ds_attr(rlay_fp, 'bounds')
            
            if ibounds == pd['bounds']:
                return rlay_fp
            else:
                ofp, stats_d = write_clip(rlay_fp,
                                  ofp=os.path.join(tmp_dir, os.path.basename(rlay_fp).replace('.tif', '_clip.tif')), 
                                  bbox=sgeo.box(*pd['bounds']), crs=pd['crs'])
                
                return ofp
            
        #=======================================================================
        # get depths
        #=======================================================================
        """doing this everytime... nice to have the wd for plots"""
        #if (sample_pts_fp!=None) or (hwm_pts_fp!=None):
        #predicted
        pred_wd_fp = get_wsh_rlay(dem_fp, pred_wse_fp, out_dir=out_dir)
        rlay_ar_apply(pred_wd_fp, assert_wsh_ar, msg='pred')
        
        self.pred_wd_fp=pred_wd_fp
        
        fp_d['pred_wd_fp'] = pred_wd_fp
            
        
        #=======================================================================
        # WD samples between grids----
        #======================================================================= 
        if (not sample_pts_fp is None) and (not true_wse_fp is None):          
            log.info(f'computing WD performance at points: \n    {sample_pts_fp}')
 
            #===================================================================
            # #build depths arrays
            #===================================================================                        
            true_wd_fp = get_wsh_rlay(dem_fp, clip_rlay(true_wse_fp), out_dir=tmp_dir)
            rlay_ar_apply(true_wd_fp, assert_wsh_ar, msg='true')
            self.true_wd_fp=true_wd_fp
            fp_d['true_wd_fp'] = true_wd_fp
            #===================================================================
            # run
            #===================================================================
            metric_lib['pts'], meta_lib['pts'] = self.run_vali_pts(sample_pts_fp,
                                        true_wd_fp=true_wd_fp, pred_wd_fp=pred_wd_fp, 
                                        **skwargs)
            
            fp_d['pts_samples_fp'] = meta_lib['pts']['samples_fp']
            
        #=======================================================================
        # HWMs--------
        #=======================================================================
        if not hwm_pts_fp is None:
            log.info(f'computing performance against HWMs ({os.path.basename(hwm_pts_fp)})')
            
            metric_lib['hwm'], meta_lib['hwm'] = self.run_vali_hwm(pred_wd_fp, hwm_pts_fp, **skwargs)
            
            fp_d['hwm_samples_fp'] = meta_lib['hwm']['samples_fp']
 
        
        #=======================================================================
        # inundatdion extents--------
        #=======================================================================
 
        #=======================================================================
        # true inundation
        #=======================================================================
        if true_inun_fp is None:
            log.info('using \'true_wse_fp\' for inundation validation')
            true_inun_fp = clip_rlay(true_wse_fp)
            
        # rasterize
        if not is_raster_file(true_inun_fp):
            log.info('rasterizing polygon')
            true_inun_rlay_fp = write_rasterize(true_inun_fp, pred_wse_fp)
            
        else:
            true_inun_rlay_fp = true_inun_fp
        
        fp_d['true_inun_rlay_fp'] = true_inun_rlay_fp
 
        #=======================================================================
        # run
        #=======================================================================
        metric_lib['inun'], confuGrid_fp = self.run_vali_inun(true_inun_fp=true_inun_rlay_fp, pred_inun_fp=pred_wse_fp, **skwargs)        
        meta_lib['inun_metrics'] = metric_lib['inun']
        fp_d['confuGrid_fp'] = confuGrid_fp
        
        #=======================================================================
        # sample with points
        #=======================================================================
        if not sample_pts_fp is None:
            fp_d['confuSamps_fp'] = self.run_vali_confuSamps(confuGrid_fp, sample_pts_fp, **skwargs)
        #=======================================================================
        # wrap-----
        #=======================================================================
        meta_lib['fps'] = fp_d
        if write_meta:
            self._write_meta(meta_lib, logger=log, out_dir=out_dir)
        
        log.info('finished')
        return metric_lib, meta_lib
    

    def _add_rel_fp(self, res_lib):
        
        for k0, v0 in res_lib.copy().items(): #simName
            for k1, v1 in v0.items(): #validation type
                if 'fp' in v1:
                    #should over-write
                    v1['fp_rel'] = {k:self._relpath(fp) for k, fp in v1['fp'].items()}

    def run_vali_multi(self,
                       pred_wsh_fp_d,
                       hwm_pts_fp=None,
                       inun_fp=None,
                       aoi_fp=None, 
                       write_pick=True,
                       write_meta=True, 
                       copy_inputs=False,                     
                       **kwargs):
        
        """build validation metrics on multiple prediction/simulated grids
        
        water depths
        HWMs and inundation extents
        
        Pars
        ----------
        inun_fp: str
            filepath to INUN_RLAY or INUN_POLY
            
        copy_inputs: bool
            copy input files over to outputs (nice for testing)
            
        Returns
        -----------
        level0: predicted WSH name
            level1: validation type (hwm, inun, raw, clip)
                level2: metric, fp, meta
                    level4: varName
                        level5: val
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rvX',  ext='.pkl', **kwargs)
        res_lib = dict()
        if aoi_fp is None: aoi_fp=self.aoi_fp
        
        #detect the inundation file type
        if is_raster_file(inun_fp):
            inun_dkey='INUN_RLAY'
        else:
            inun_dkey='INUN_POLY'
        
        log.info(f'on {len(pred_wsh_fp_d)}/n    {pred_wsh_fp_d.keys()}')
        #=======================================================================
        # precheck
        #=======================================================================
        HydTypes(inun_dkey).assert_fp(inun_fp)
 
        for k,fp in pred_wsh_fp_d.items():
            HydTypes('WSH').assert_fp(fp,msg=k)
 
            
        #=======================================================================
        # clip
        #=======================================================================
        #set the raw
        pred_wsh_fp_d_raw = pred_wsh_fp_d.copy()
        pred_wsh_fp_d_clip = None
        inun_fp_raw=inun_fp

        if not aoi_fp is None:
            bbox, crs = get_bbox_and_crs(aoi_fp)
            
            #predicted rasters
            clip_d = self.clip_rlays(pred_wsh_fp_d, bbox=bbox, crs=crs)
            pred_wsh_fp_d_clip={k:v['clip_fp'] for k,v in clip_d.items()}
            pred_wsh_fp_d = pred_wsh_fp_d_clip.copy()
            
 
        
            #inundation raster
            if is_raster_file(inun_fp):
                bnm = os.path.splitext(os.path.basename(inun_fp))[0]
                log.debug(f'clipping inundation raster {bnm}')
                
                inun_fp, _ = write_clip(inun_fp, bbox=bbox, ofp=os.path.join(out_dir, f'{bnm}_clip.tif'))
 
        
        #=======================================================================
        # precheck
        #=======================================================================
        
        vlast = None
        for k,v in pred_wsh_fp_d.items():
            rlay_ar_apply(v, assert_wsh_ar, msg=k)
            
            #extents
            """allowing differnt resolutions.. but extents must match"""
            if not vlast is None:
                assert_extent_equal(vlast, v, msg=f'{v} extent mismatch')
            vlast = v
 
        #=======================================================================
        # prep the inundation library
        #=======================================================================
        if not inun_fp is None:
            if is_raster_file(inun_fp):
                assert_extent_equal(vlast, inun_fp, msg=f'{v} extent mismatch')
                self.inun_rlay_fp_lib['raw'] = inun_fp #add this entry
        
        #=======================================================================
        # loop and compute on each
        #=======================================================================
 
        log.info(f'computing performance of {len(pred_wsh_fp_d)}')
        for k, wsh_fp in pred_wsh_fp_d.items():
            
            #===================================================================
            # setup this observation
            #===================================================================            
            logi = log.getChild(k)
            logi.info(f'on {k}: {os.path.basename(wsh_fp)}\n\n')
            rdi = dict()
            
            odi = os.path.join(out_dir, k)
            if not os.path.exists(odi):os.makedirs(odi)
            
            skwargs = dict(logger=logi, resname=k, out_dir=odi, tmp_dir=tmp_dir, subdir=False)
            
            #=======================================================================
            # HWMs--------
            #=======================================================================
            if not hwm_pts_fp is None:
                d=dict()                
                d['metric'], d['fp'], d['meta']  = self.run_vali_hwm(wsh_fp, hwm_pts_fp, **skwargs)
                rdi['hwm']=d
                
            #===================================================================
            # inundation-------
            #===================================================================
            if not inun_fp is None:
                d=dict()
                #get the observed inundation (transform)
                inun_rlay_fp = self._get_inun_rlay(inun_fp, wsh_fp, **skwargs)
 
                
                
                #convert the wsh to binary inundation
                pred_inun_fp = write_wsh_boolean(wsh_fp)
                assert_spatial_equal(inun_rlay_fp, pred_inun_fp, msg=k)
                
                #run the validation                
                d['metric'], d['fp'], d['meta'] = self.run_vali_inun(
                    true_inun_fp=inun_rlay_fp, pred_inun_fp=pred_inun_fp, **skwargs) 
                
                rdi['inun']=d
                
            #===================================================================
            # append inputs to results pick
            #===================================================================
            def c(fp): 
                if copy_inputs:
                    ofp = os.path.join(odi,os.path.basename(fp)) 
                    return copyr(fp, ofp)
                else:
                    return fp
 
 
            #add inputs
            rdi['raw'] = {'fp':{k:c(v) for k,v in {'wsh':pred_wsh_fp_d_raw[k], 'inun':inun_fp_raw}.items()}}
            
            if not pred_wsh_fp_d_clip is None:
                rdi['clip'] ={'fp':{k:c(v) for k,v in {'wsh':pred_wsh_fp_d_clip[k], 'aoi':aoi_fp, 'inun':inun_fp}.items()}}
                #rdi['clip'] = {'fp':}
            
            #===================================================================
            # wrap
            #===================================================================
            assert len(rdi)>0
            log.debug(f'finished on {k} w/ {len(rdi)}')
            #print(dstr(rdi))
            res_lib[k] = rdi
 
        #=======================================================================
        # add relatives
        #=======================================================================
        log.debug('adding relative paths')
        self._add_rel_fp(res_lib) 
        #print(dstr(res_lib))        
            
        #=======================================================================
        # write meta and pick
        #=======================================================================
        
        if write_meta:
            self._write_meta_vali(res_lib)
 
        if write_pick: 
            assert not os.path.exists(ofp)
            with open(ofp,'wb') as file:
                pickle.dump(res_lib, file)
 
            log.info(f'wrote res_lib pickle to \n    {ofp}')
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished on {len(res_lib)}')
        
        return res_lib
                
                
    def _get_inun_rlay(self,
                       raw_fp,
                       ref_fp,
                       **kwargs):
        """intelligent retrival and preparation of an inundation raster
        
        expecting binary rasters
        
        Returns
        --------
        str
            filepath to inundation raster (0=wet)
        """
        log, tmp_dir, out_dir, _, resname = self._func_setup('inunR',  **kwargs)
        
        
        
        #=======================================================================
        # see if we already have a nice inundation raster
        #=======================================================================        
        """the raw should already be in teh library"""
        inun_fp=None
        log.debug(f'scanning {len(self.inun_rlay_fp_lib)} for a match')
        for k, rlay_fpi in self.inun_rlay_fp_lib.items():
            if is_spatial_equal(ref_fp, rlay_fpi):
                inun_fp = rlay_fpi
                log.info(f'found inundation raster match in {k}')
                
        #=======================================================================
        # generate a new one
        #=======================================================================
        if inun_fp is None:
            #===================================================================
            # setup
            #===================================================================
            spatial_meta_d = get_meta(ref_fp)
            d = {k:v for k,v in spatial_meta_d.items() if k in ['crs', 'height', 'width']}
            log.info(f'building an matching inundation raster from {os.path.basename(raw_fp)} to match:\n    {d}')
            fname = f'inun_%ix%i'%(d['height'], d['width'])
            ofp = os.path.join(out_dir, fname+'.tif')
            #===================================================================
            # rasterize
            #===================================================================
 
            if not is_raster_file(raw_fp):
                log.info(f'rasterizing inundation polygon from {os.path.basename(raw_fp)}')
                inun1_fp = write_rasterize(raw_fp, ref_fp, out_dir=tmp_dir)
                
                #convert to boolean mask
                inun_fp = write_extract_mask(inun1_fp, maskType='binary', invert=True, ofp=ofp)
                
            #===================================================================
            # resample
            #===================================================================
            else:
                raise NotImplementedError('dome')
                sup_rat = get_support_ratio(raw_fp, ref_fp)
                
 
                
            #check
            assert_mask_fp(inun_fp, maskType='binary')
            rlay_mar_apply(inun_fp, assert_partial_wet, maskType='binary', msg=os.path.basename(raw_fp))
            #===================================================================
            # update lib
            #===================================================================
            self.inun_rlay_fp_lib[fname] = inun_fp
            
        
        HydTypes('INUN_RLAY').assert_fp(inun_fp)
 
        return inun_fp
 

    def _write_meta_vali(self, res_lib):
        #collect
 
        meta_lib = dict()
        for k0, v0 in res_lib.items(): #simName
            for k1, v1 in v0.items(): #validation type
                if 'meta' in v1:
                    meta_lib[f'{k0}.{k1}'] = v1['meta']
 
        #write
        
        meta_lib = {**{'init':self._get_init_pars()}, **meta_lib}
        self._write_meta(meta_lib)
    

def run_validator(run_kwargs=dict(),init_kwargs=dict()):
    """compute error metrics and layers on a wse layer
    
    """
    
    with ValidateSession(**init_kwargs) as ses:
        return ses.run_vali(**run_kwargs)
 

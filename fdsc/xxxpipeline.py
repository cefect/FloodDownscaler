'''
Created on Jan. 7, 2023

@author: cefect

integrated downscaling and validation
'''

import os, datetime, pickle, pprint
import shapely.geometry as sgeo
import rasterio as rio
from rasterio.enums import Resampling, Compression

from definitions import wrk_dir, src_name


from hp.basic import today_str
from hp.rio import (
    write_clip,assert_spatial_equal,assert_extent_equal,write_resample,is_raster_file,
    )

from hp.hyd import get_wsh_rlay


from fdsc.scripts.control import Dsc_Session, nicknames_d
from fdsc.analysis.valid.v_ses import ValidateSession


def now():
    return datetime.datetime.now()

#===============================================================================
# class--------
#===============================================================================
class PipeSession(Dsc_Session, ValidateSession):
    
    def __init__(self,
                 run_name=None,
                 **kwargs):
 
        if run_name is None:
            run_name = 'pipe'
        super().__init__(run_name=run_name, **kwargs)

    def clip_set(self, raster_fp_d, aoi_fp=None, bbox=None, crs=None,
                 sfx='clip', **kwargs):
        """clip a dicrionary of raster filepaths
        
        Parameters
        -----------
        raster_fp_d: dict
            {key:filepath to raster}
            
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('clip_set', subdir=True, **kwargs) 
     
        #=======================================================================
        # retrive clipping parameters
        #=======================================================================
        if not aoi_fp is None:
            self._set_aoi(aoi_fp=aoi_fp)
            
        write_kwargs =self._get_defaults(bbox=bbox, crs=crs, as_dict=True)
        bbox = write_kwargs['bbox'] #for reporting
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(raster_fp_d, dict)
        
        assert isinstance(bbox, sgeo.Polygon)
        assert hasattr(bbox, 'bounds')
        
        #=======================================================================
        # clip each
        #=======================================================================
        log.info(f'clipping {len(raster_fp_d)} rasters to \n    {bbox}')
        res_d = dict()
        for key, fp in raster_fp_d.items():
            d={'og_fp':fp}
            d['clip_fp'], d['stats'] = write_clip(fp,ofp=os.path.join(out_dir, f'{key}_{sfx}.tif'), 
                                                  **write_kwargs)
            
            res_d[key] = d
            
        log.info(f'finished on {len(res_d)}')
        return res_d
        


    def get_depths_coarse(self, wse2_fp, dem1_fp, downscale=None, **kwargs):
        """get the coarse depths"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dep2', subdir=True, **kwargs)
        if downscale is None:
            downscale = self.get_downscale(wse2_fp, dem1_fp)
        
        meta_d = {'downscale':downscale, 'wse2_fp':wse2_fp, 'dem_fp':dem1_fp}
        #=======================================================================
        # #upscale DEM
        #=======================================================================
        log.info('building depths grid')
        dem2_fp = write_resample(dem1_fp, resampling=Resampling.bilinear, 
            scale=1 / downscale, out_dir=tmp_dir)
        #write depths
        meta_d['dep2'] = get_wsh_rlay(dem2_fp, wse2_fp,ofp=ofp)
        
        return ofp, meta_d


    def _wrap_meta(self, meta_lib, **kwargs):
        tdelta = (now() - meta_lib['smry']['start']).total_seconds()
        meta_lib['smry']['tdelta'] = tdelta
        #collapse and promote
        md = dict()
        for k0, d0 in meta_lib.items():
            d0m = dict()
            for k1, d1 in d0.items():
                #promte contents
                if isinstance(d1, dict):
                    md[k0 + '_' + k1] = d1
                else:
                    d0m[k1] = d1
            
            if len(d0m) > 0:
                md[k0] = d0m
        
    #write
        _ = self._write_meta(md, **kwargs)
        
        return meta_lib

    def run_dsc_vali(self,
            wse2_fp,
            dem1_fp,
            
            aoi_fp=None,
            
            dsc_kwargs=dict(method = 'CostGrow'),
            vali_kwargs=dict(), 
            **kwargs
            ):
        """generate downscale then compute metrics (one method) 
        
        Parameters
        ------------
 
            
        kwargs: dict
            catchAll... including any method-specific parameters
        """
        
        #===========================================================================
        # defaults
        #===========================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rPipe', subdir=False, **kwargs)
 
        meta_lib = {'smry':{**{'today':self.today_str, 'aoi_fp':aoi_fp, 'start':now()}, **self._get_init_pars()}}
        
        if aoi_fp is None: aoi_fp=self.aoi_fp
        skwargs = dict(out_dir=out_dir, tmp_dir=tmp_dir)            
        
        #=======================================================================
        # precheck
        #=======================================================================
        #assert_spatial_equal(dem1_fp, wse1V_fp, msg='DEM and validation')
        assert_extent_equal(dem1_fp, wse2_fp, msg='DEM and WSE')
        
        #=======================================================================
        # helpers
        #=======================================================================
        write_pick = lambda obj, dkey: self.write_pick(obj, dkey, out_dir=out_dir, log=log)
        
        #=======================================================================
        # prelims
        #=======================================================================
        downscale = self.get_downscale(wse2_fp, dem1_fp)
        meta_lib['smry']['downscale'] = downscale
        dsc_kwargs['downscale'] = downscale
        log.info(f'downscale={downscale}')
        #=======================================================================
        # clip raw rasters
        #=======================================================================
        fp_d = {'wse2':wse2_fp, 'dem1':dem1_fp}        
        if not aoi_fp is None:            
            clip_fp_d = self.clip_set(fp_d, aoi_fp=aoi_fp, **skwargs)            
            d = {k:v['clip_fp'] for k,v in clip_fp_d.items()}            
        else:
            d = fp_d
        
        wse2_fp, dem1_fp = d['wse2'], d['dem1']
        meta_lib['smry'].update(d)  #add cropped layers to summary
        
        #=======================================================================
        # downscale------
        #=======================================================================
        wse1_fp, meta_lib['dsc'] = self.run_dsc(wse2_fp,dem1_fp,write_meta=True,
                                               #out_dir=os.path.join(ses.out_dir, 'dsc'), 
                                               ofp=self._get_ofp('dsc'),logger=log, **dsc_kwargs)
 
        meta_lib['smry']['wse1'] = wse1_fp #promoting key results to the summary page
        
        #=======================================================================
        # validate-------
        #=======================================================================
        metric_lib, meta_lib['vali'] = self.run_vali(pred_wse_fp=wse1_fp, dem_fp=dem1_fp,  
                                                     write_meta=True, logger=log,
                                                    #out_dir=os.path.join(ses.out_dir, 'vali'), 
                                                    **vali_kwargs)
        
 
        meta_lib['smry']['valiMetrics_fp'] = write_pick(metric_lib, 'valiMetrics')
        
        #=======================================================================
        # get depths-------
        #=======================================================================
        """same for all algos... building for consistency"""
        dep2_fp, meta_lib['dep2'] = self.get_depths_coarse(wse2_fp, dem1_fp, downscale=downscale,**skwargs)
        
        meta_lib['smry']['dep2'] = dep2_fp #promoting key results to the summary page

        #=======================================================================
        # meta
        #=======================================================================
        meta_lib = self._wrap_meta(meta_lib, logger=log)

        meta_fp = write_pick(meta_lib, 'meta_lib')
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished in %s at \n    {out_dir}'%meta_lib['smry']['tdelta'])
            
        return meta_fp
    
    def run_hyd_vali(self,
            wse_fp,
            dem1_fp,
            
            aoi_fp=None,
            
 
            vali_kwargs=dict(), 
            **kwargs
            ):
        """validation and depths building pipeline for hydro rasters
        
        Parameters
        ------------
 
            
        kwargs: dict
            catchAll... including any method-specific parameters
        """
        
        #===========================================================================
        # defaults
        #===========================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rhv', subdir=False, **kwargs)
 
        
        if aoi_fp is None: aoi_fp=self.aoi_fp
 
        meta_lib = {'smry':{**{'today':self.today_str, 'aoi_fp':aoi_fp, 'start':now()}, **self._get_init_pars()}}
        skwargs = dict(out_dir=out_dir, tmp_dir=tmp_dir)
        log.info(f'validating \'{os.path.basename(wse_fp)}\'')
        #=======================================================================
        # helpers
        #=======================================================================
        write_pick = lambda obj, dkey: self.write_pick(obj, dkey, out_dir=out_dir, log=log)
        
 
        
        #=======================================================================
        # clip raw rasters
        #=======================================================================
        fp_d = {'wse':wse_fp, 'dem1':dem1_fp}        
        if not aoi_fp is None:            
            clip_fp_d = self.clip_set(fp_d, aoi_fp=aoi_fp, logger=log, **skwargs)            
            d = {k:v['clip_fp'] for k,v in clip_fp_d.items()}            
        else:
            d = fp_d
        
        wse_fp, dem1_fp = d['wse'], d['dem1']
        meta_lib['smry'].update(d)  #add cropped layers to summary
        
        #=======================================================================
        # rescale
        #=======================================================================
 
        downscale = self.get_resolution_ratio(wse_fp, dem1_fp)
        
        if not downscale == 1: 
            wse1_fp = write_resample(wse_fp, scale=downscale, resampling=Resampling.nearest, out_dir=out_dir)
        else:
            wse1_fp = wse_fp
        
        #=======================================================================
        # validate-------
        #=======================================================================
        metric_lib, meta_lib['vali'] = self.run_vali(pred_wse_fp=wse1_fp, dem_fp=dem1_fp,  
                                                     write_meta=True,  logger=log,
                                                    **vali_kwargs)
        
 
        meta_lib['smry']['valiMetrics_fp'] = write_pick(metric_lib, 'valiMetrics')
        
        #=======================================================================
        # get depths-------
        #=======================================================================
        #=======================================================================
        # """same for all algos... building for consistency"""
        # dep2_fp, meta_lib['dep2'] = self.get_depths_coarse(wse2_fp, dem1_fp, downscale=downscale,**skwargs)
        # 
        # meta_lib['smry']['dep2'] = dep2_fp #promoting key results to the summary page
        #=======================================================================
        
        #=======================================================================
        # meta
        #=======================================================================
        meta_lib = self._wrap_meta(meta_lib, logger=log)

        meta_fp = write_pick(meta_lib, 'meta_lib')
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished in %s at \n    {out_dir}'%meta_lib['smry']['tdelta'])
            
        return meta_fp
        
          
        
    def write_pick(self, obj, sfx, out_dir=None, log=None):
        ofpi = self._get_ofp(out_dir=out_dir, dkey=sfx, ext='.pkl')
        with open(ofpi,  'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        log.info(f'wrote \'{sfx}\' {type(obj)} to \n    {ofpi}')
        return ofpi
         
 
def run_pipeline_multi(
        
          wse2_fp=None, 
          dem1_fp=None,
          vali_kwargs=dict(),
        method_pars={'CostGrow': {}, 
                     'Basic': {}, 
                     'SimpleFilter': {}, 
                     'BufferGrowLoop': {}, 
                     'Schumann14': {},
                     },
        
        validate_hyd=True,
 
        logger=None,
           
 
        **kwargs):
    """run the pipeline on a collection of methods
    
    Pars
    ------
    method_pars: dict
        method name: kwargs
        
    vali_kwargs: dict
        kwargs for validation. see fdsc.analysis.valid.v_ses.ValidateSession.run_vali()
        
    validate_hyd: bool
        whether to validate the hyd wse rasters also
                
    """
    
 
    res_d = dict()
 
    #===========================================================================
    # loop onmethods
    #===========================================================================        
    for method, mkwargs in method_pars.items():
        assert method in nicknames_d, f'method {method} not recognized\n    {list(nicknames_d.keys())}'
        name = nicknames_d[method] 
        print(f'\n\nMETHOD={method}\n\n')
            
        #=======================================================================
        # run on session
        #=======================================================================
        """want a clean session for each method"""
        with PipeSession(logger=logger, run_name=name, **kwargs) as ses:            
            #===================================================================
            # defaults
            #===================================================================            
            skwargs = dict(out_dir=ses.out_dir, logger=ses.logger.getChild(name))            
            #===================================================================
            # run
            #===================================================================
            res_d[method] = ses.run_dsc_vali(wse2_fp, dem1_fp,
                                dsc_kwargs=dict(method=method, rkwargs=mkwargs),
                                vali_kwargs=vali_kwargs,
                                **skwargs) 
            
            #===================================================================
            # passthrough
            #===================================================================
 
            logger = ses.logger
            
    #===========================================================================
    # validate hydrodyn rasetrs
    #===========================================================================
    if validate_hyd:        
        
        #extract kwargs of interest
        vali_kwargs2 = {k:v for k,v in vali_kwargs.items() if k in ['true_inun_fp', 'sample_pts_fp', 'hwm_pts_fp']}
        
        #=======================================================================
        # run on each hydro result
        #=======================================================================
        for name, wse_fp in {'WSE2':wse2_fp, 'WSE1':vali_kwargs['true_wse_fp']}.items():
            print(f'\n\n HYDRO VALI on {name}\n\n')
 
            with PipeSession(logger=logger, run_name=name.lower()+'_vali', **kwargs) as ses:
                skwargs = dict(out_dir=ses.out_dir, logger=ses.logger.getChild(name)) 
                
                res_d[name] = ses.run_hyd_vali(wse_fp, dem1_fp,vali_kwargs=vali_kwargs2, **skwargs)
 
 
    print('finished on \n    ' + pprint.pformat(res_d, width=30, indent=True, compact=True, sort_dicts =False))
    return res_d
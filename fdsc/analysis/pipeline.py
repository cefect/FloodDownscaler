'''
Created on Jan. 7, 2023

@author: cefect

integrated downscaling and validation
'''

import os, datetime, pickle
import shapely.geometry as sgeo
import rasterio as rio
from rasterio.enums import Resampling, Compression

from definitions import wrk_dir, src_name


from hp.basic import today_str
from hp.rio import (
    write_clip,assert_spatial_equal,assert_extent_equal,get_depth,write_resample,
    )


from fdsc.scripts.dsc import Dsc_Session
from fdsc.analysis.valid import ValidateSession


def now():
    return datetime.datetime.now()

#===============================================================================
# class--------
#===============================================================================
class PipeSession(Dsc_Session, ValidateSession):

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
        

def run_dsc_vali(
        wse2_rlay_fp,
        dem1_rlay_fp,
        wse1V_fp=None,
        dsc_kwargs=dict(dryPartial_method = 'costGrowSimple'),
        vali_kwargs=dict(),
 
        **kwargs
        ):
    """generate downscale then compute metrics (one option)
    
    Parameters
    ------------
    wse1_V_fp: str
        filepath to wse1 raster (for validation)
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    with PipeSession(logfile_duplicate=True, dem_fp=dem1_rlay_fp, **kwargs) as ses:
        start = now()
        log = ses.logger.getChild('r')
        meta_lib = {'smry':{**{'today':ses.today_str}, **ses._get_init_pars()}}
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert_spatial_equal(dem1_rlay_fp, wse1V_fp, msg='DEM and validation')
        assert_extent_equal(dem1_rlay_fp, wse2_rlay_fp, msg='DEM and WSE')
        
        #=======================================================================
        # helpers
        #=======================================================================
        def write(obj, sfx):
            ofpi = ses._get_ofp(out_dir=ses.out_dir, dkey=sfx, ext='.pkl')
            with open(ofpi,  'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            log.info(f'wrote \'{sfx}\' {type(obj)} to \n    {ofpi}')
            return ofpi
        #=======================================================================
        # clip raw rasters
        #=======================================================================
        fp_d = {'wse2':wse2_rlay_fp, 'dem1':dem1_rlay_fp, 'wse1V': wse1V_fp}
        
        
        if not ses.aoi_fp is None:
            assert not wse1V_fp is None, 'not implemented'             
            clip_fp_d = ses.clip_set(fp_d)            
            d = {k:v['clip_fp'] for k,v in clip_fp_d.items()}            
        else:
            d = fp_d
        
        wse2_fp, dem1_fp, wse1V_fp = d['wse2'], d['dem1'], d['wse1V']
        meta_lib['smry'].update(fp_d)  
        #=======================================================================
        # downscale------
        #=======================================================================
        wse1_fp, meta_lib['dsc'] = ses.run_dsc(wse2_fp,dem1_fp,write_meta=True, 
                                               ofp=ses._get_ofp('dsc'), **dsc_kwargs)
 
        meta_lib['smry']['wse1'] = wse1_fp #promoting key results to the summary page
        #=======================================================================
        # validate-------
        #=======================================================================
        metric_lib, meta_lib['vali'] = ses.run_vali(true_fp=wse1V_fp, pred_fp=wse1_fp, dem_fp=dem1_fp, write_meta=True, **vali_kwargs)
        
 
        meta_lib['smry']['valiMetrics_fp'] = write(metric_lib, 'valiMetrics')
        
        #=======================================================================
        # get depths-------
        #=======================================================================
        """nice for some plots"""
        #upscale DEM        
        log.info('building depths grid')
        dem2_fp = write_resample(dem1_rlay_fp, resampling=Resampling.bilinear, scale=1/ses.downscale,out_dir=ses.tmp_dir)
  
        #write depths
        meta_lib['smry']['dep2'] = get_depth(dem2_fp, wse2_rlay_fp, 
                                         ofp=ses._get_ofp(out_dir=ses.out_dir, dkey='dep2'))
        
        #=======================================================================
        # meta
        #=======================================================================
        print(meta_lib.keys())
        #collapse and promote
        md=dict()
        for k0, d0 in meta_lib.items():
            
            d0m = dict()
            for k1, d1 in d0.items():
               
                #promte contents
                if isinstance(d1, dict):
                    md[k0+'_'+k1] = d1
                else:
                    d0m[k1]=d1
                    
            if len(d0m)>0:
                md[k0]=d0m
                
        #write 
        meta_fp  = ses._write_meta(md, logger=log)
        write(meta_lib, 'meta_lib')

        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished in {now()-start} at \n    {ses.out_dir}')
 

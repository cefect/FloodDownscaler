'''
Created on Jan. 7, 2023

@author: cefect

integrated downscaling and validation
'''

import os, datetime
import shapely.geometry as sgeo

from definitions import wrk_dir, src_name


from hp.basic import today_str
from hp.rio import (
    write_clip,
    )


from fdsc.scripts.dsc import Dsc_Session
from fdsc.scripts.valid import ValidateSession


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
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('clip_set',  **kwargs) 
     
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
        dsc_kwargs=dict(dryPartial_method = 'costDistanceSimple'),
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
    with PipeSession(**kwargs) as ses:
        start = now()
        log = ses.logger.getChild('r')
        
        #=======================================================================
        # clip rasters
        #=======================================================================

        fp_d = {'wse2':wse2_rlay_fp, 'dem1':dem1_rlay_fp, 'wse1V': wse1V_fp}
        
        
        if not ses.aoi_fp is None:
            assert not wse1V_fp is None, 'not implemented'
             
            clip_fp_d = ses.clip_set(fp_d)
            
            d = {k:v['clip_fp'] for k,v in clip_fp_d.items()}            
        else:
            d = fp_d
        
        wse2_fp, dem1_fp, wse1V_fp = d['wse2'], d['dem1'], d['wse1V']
            
        #=======================================================================
        # downscale
        #=======================================================================
        wse1_fp = ses.run_dsc(wse2_fp,dem1_fp,**dsc_kwargs)
 
        #=======================================================================
        # validate
        #=======================================================================
        _ = ses.run_vali(true_fp=wse1V_fp, pred_fp=wse1_fp, **vali_kwargs)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished in {now()-start}')
 

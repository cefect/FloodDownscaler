'''
Created on Mar. 27, 2023

@author: cefect

running evaluation on downscaling results
'''
import os

from hp.basic import dstr
from hp.fiona import get_bbox_and_crs
from hp.hyd import assert_type_fp
from hp.rio import write_clip, is_raster_file

from fdsc.base import DscBaseSession, assert_dsc_res_lib
from fperf.pipeline import ValidateSession

class Dsc_Eval_Session(DscBaseSession, ValidateSession):
    """special methods for validationg downscample inundation rasters"""
    
    def _get_fps_from_dsc_lib(self,
                             dsc_res_lib,
                             relative=None, base_dir=None,
                             **kwargs):
        """extract paramter container for post from dsc results formatted results library"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('gfps',  **kwargs)
        if relative is None: relative=self.relative
        if base_dir is None: base_dir=self.base_dir
        
        #=======================================================================
        # precheck
        #=======================================================================
        assert_dsc_res_lib(dsc_res_lib)
        
        log.info(f'on {len(dsc_res_lib)} w/ relative={relative}')
        #=======================================================================
        # extract
        #=======================================================================\
        res_d = dict()
        for k0, d0 in dsc_res_lib.items():
            #select the 
            if relative:
                fp_d = d0['fp_rel']
                res_d[k0] = {k:os.path.join(base_dir, v) for k,v in fp_d.items() if not '_raw' in k}
            else:
                res_d[k0]=d0['fp']
                
        

        #=======================================================================
        # check
        #=======================================================================
        for simName, d0 in res_d.items():
            for k, v in d0.items():
                assert os.path.exists(v), f'{simName}.{k}\n    {v}'
                
        log.debug('\n'+dstr(res_d))
        
        return res_d
    
    def run_vali_multi_dsc(self,
                           fp_lib,
                           aoi_fp=None,
                           vali_kwargs=dict(),
                           **kwargs):
        """skinny wrapper for test_run_vali_multi using dsc formatted results"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('gfps',  **kwargs)
        if aoi_fp is None: aoi_fp=self.aoi_fp
        
        #=======================================================================
        # precheck
        #=======================================================================
        
        assert set(fp_lib.keys()).difference(self.nicknames_d.keys())==set(['inputs'])
        
        #=======================================================================
        # separate and type check
        #=======================================================================
        
        ins_d = fp_lib.pop('inputs')
        dem_fp = ins_d['DEM']
        assert_type_fp(dem_fp, 'DEM')
        
        #pull WSE rasters
        pred_wse_fp_d = {k:v['WSE1'] for k,v in fp_lib.items()}
        for k,fp in pred_wse_fp_d.items():
            assert_type_fp(fp, 'WSE', msg=k)
        
        
        #=======================================================================
        # clip
        #=======================================================================
        #set the raw
        ins_raw_d = {'DEM':dem_fp}.update(pred_wse_fp_d)

        if not aoi_fp is None:
            bbox, crs = get_bbox_and_crs(aoi_fp)
            
            #predicted rasters
            clip_d = self.clip_rlays(pred_wse_fp_d, bbox=bbox, crs=crs)
            pred_wse_fp_d={k:v['clip_fp'] for k,v in clip_d.items()}
            #pred_wse_fp_d = pred_wse_fp_d_clip.copy()
            
            dem_fp = write_clip(dem_fp,bbox=bbox,crs=crs,
                                ofp=os.path.join(out_dir, f'{os.path.basename(dem_fp)}_clip.tif')
                                                  )
        
        ins_d = {'DEM':dem_fp}.update(pred_wse_fp_d)
            
 
        
        #=======================================================================
        #     #inundation raster
        #     if is_raster_file(inun_fp):
        #         bnm = os.path.splitext(os.path.basename(inun_fp))[0]
        #         log.debug(f'clipping inundation raster {bnm}')
        #         
        #         inun_fp, _ = write_clip(inun_fp, bbox=bbox, ofp=os.path.join(out_dir, f'{bnm}_clip.tif'))
        # 
        #=======================================================================
        #=======================================================================
        # get WSH
        #=======================================================================
        log.debug(f'on \n%s'%dstr(pred_wse_fp_d))
        
        
        
                
 
                
            
                             
 
        
 
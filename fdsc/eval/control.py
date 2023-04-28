'''
Created on Mar. 27, 2023

@author: cefect

running evaluation on downscaling results
'''
import os, pickle, shutil

import rasterio.shutil

from hp.basic import dstr
from hp.fiona import get_bbox_and_crs
from hp.hyd import get_wsh_rlay
from hp.rio import write_clip, is_raster_file, copyr

from fdsc.base import assert_dsc_res_lib, DscBaseWorker, assert_type_fp
#from fdsc.control import Dsc_Session
from fperf.pipeline import ValidateSession
 

class Dsc_Eval_Session(DscBaseWorker, ValidateSession):
    """special methods for validationg downscample inundation rasters"""
    
    def _get_fps_from_dsc_lib(self,
                             dsc_res_lib,
                             relative=None, base_dir=None,
                             **kwargs):
        """extract paramter container for post from dsc results formatted results library"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('gfps',  **kwargs)
        if relative is None: relative=self.relative
        if base_dir is None: base_dir=self.base_dir
        assert os.path.exists(base_dir)
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
                assert os.path.exists(v), f'bad file on {simName}.{k}\n    {v}'
                
        log.debug('\n'+dstr(res_d))
        
        return res_d
    



    def run_vali_multi_dsc(self,
                           fp_lib,
                           #aoi_fp=None,
                           vali_kwargs=dict(),
                           write_meta=True,
                           write_pick=True,
                           copy_inputs=False,  
                           **kwargs):
        """skinny wrapper for test_run_vali_multi using dsc formatted results"""
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rvmd', ext='.pkl', **kwargs)
        #if aoi_fp is None: aoi_fp=self.aoi_fp
        
        #=======================================================================
        # PRE------
        #=======================================================================
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
        """even though it would be nicer to only build WSH on the clip
            easier to keep it on the sub func"""
#===============================================================================
#         #set the raw
#         ins_raw_d = {'DEM':dem_fp}.update(pred_wse_fp_d)
# 
#         if not aoi_fp is None:
#             bbox, crs = get_bbox_and_crs(aoi_fp)
#             
#             #predicted rasters
#             clip_d = self.clip_rlays(pred_wse_fp_d, bbox=bbox, crs=crs)
#             pred_wse_fp_d={k:v['clip_fp'] for k,v in clip_d.items()}
#             #pred_wse_fp_d = pred_wse_fp_d_clip.copy()
#             
#             dem_fp = write_clip(dem_fp,bbox=bbox,crs=crs,
#                                 ofp=os.path.join(out_dir, f'{os.path.basename(dem_fp)}_clip.tif')
#                                                   )
#         
#         ins_d = {'DEM':dem_fp}.update(pred_wse_fp_d)
#             
#  
#         
#         #=======================================================================
#         #     #inundation raster
#         #     if is_raster_file(inun_fp):
#         #         bnm = os.path.splitext(os.path.basename(inun_fp))[0]
#         #         log.debug(f'clipping inundation raster {bnm}')
#         #         
#         #         inun_fp, _ = write_clip(inun_fp, bbox=bbox, ofp=os.path.join(out_dir, f'{bnm}_clip.tif'))
#         # 
#         #=======================================================================
#===============================================================================
        #=======================================================================
        # get WSH
        #=======================================================================
        wsh_fp_d = dict()
        log.debug(f'on \n%s'%dstr(pred_wse_fp_d))
        for k, wse_fp in pred_wse_fp_d.items():
            fnm = os.path.splitext(os.path.basename(wse_fp))[0]
            odi = os.path.join(out_dir, k)
            if not os.path.exists(odi):os.makedirs(odi)
            wsh_fp_d[k] = get_wsh_rlay(dem_fp, wse_fp, 
                             ofp=os.path.join(odi, f'{fnm}_WSH.tif'),
                             )
            
        #=======================================================================
        # RUN----------
        #=======================================================================
 
        res_lib = self.run_vali_multi(wsh_fp_d, **vali_kwargs,logger=log, out_dir=out_dir,
                                      write_meta=False,  write_pick=False, copy_inputs=copy_inputs)
        
        #=======================================================================
        # POST---------
        #=======================================================================
        print(dstr(res_lib))
        
        
        
        #=======================================================================
        # add WSE and DEMback into results
        #=======================================================================
        #=======================================================================
        # odi=os.path.join(out_dir, 'inputs')
        # if not os.path.exists(odi):os.makedirs(odi)
        #=======================================================================
 
        def c(fp, odi): 
            """copy helper
            run_vali_multi copies inputs over to out_dir/methodName
            adds this to res_lib under level1 (see above)
            """
            if copy_inputs:
                """nice to add WSE if this isn't included"""
                fname = os.path.splitext(os.path.basename(fp))[0]
                if not 'WSE' in fname:                    
                    ofp = os.path.join(odi, f'{fname}_WSE.tif')
                else:
                    ofp = os.path.join(odi,os.path.basename(fp))
 
                 
                return copyr(fp, ofp)
            else:
                return fp
            
        #DEM
        if copy_inputs:
            dem_fp1 = os.path.join(out_dir, os.path.basename(dem_fp))
            copyr(dem_fp, dem_fp1)
        else:
            dem_fp1 = dem_fp
            
        #WSE
        log.debug('adding wse')
        for k0, v0 in res_lib.items(): #simName
            for k1, v1 in v0.items():#validation type
                if  k1=='raw':
                    #print(dstr(v1['fp'])) 
 
                    v1['fp']['wse'] = c(pred_wse_fp_d[k0], os.path.dirname(list(v1['fp'].values())[0])) #note htese aren't clipped
                    v1['fp']['dem'] = dem_fp1
                    #print(v1['fp'].keys())
                    
        
        
        for k0, v0 in res_lib.items():
            log.debug(f'{k0}\n'+dstr(v0['raw']['fp']))
 
        #=======================================================================
        # add relative paths
        #=======================================================================

                
        log.debug(f'adding relative paths w/ relative={self.relative}, base_dir={self.base_dir}')
        for k0, v0 in res_lib.copy().items(): #simName
            for k1, v1 in v0.items():#validation type
                if 'fp' in v1: 
                    #should over-write                       
                    v1['fp_rel'] =  {k:self._relpath(fp) for k, fp in v1['fp'].items()}
                                   
 
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
        log.info(f'finished on \n    {res_lib.keys()}')
        return res_lib
            
        
        
        
        
        
        
                
 
                
            
                             
 
        
 
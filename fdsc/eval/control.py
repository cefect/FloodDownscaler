'''
Created on Mar. 27, 2023

@author: cefect

running evaluation on downscaling results
'''
import os, pickle, shutil

import rasterio.shutil

from hp.basic import dstr
from hp.fiona import get_bbox_and_crs
from hp.hyd import get_wsh_rlay, HydTypes
from hp.rio import (
    write_clip, is_raster_file, copyr, assert_extent_equal, assert_spatial_equal,
    )
from hp.riom import write_extract_mask

from fdsc.base import assert_dsc_res_lib, DscBaseWorker, assert_type_fp
from fdsc.control import Dsc_Session_skinny
from fperf.pipeline import ValidateSession
 

class Dsc_Eval_Session(ValidateSession, Dsc_Session_skinny):
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
        
        #print(dstr(dsc_res_lib))
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
                           hwm_pts_fp=None,
                           inun_fp=None,
                           write_meta=True,
                           write_pick=True,
                           copy_inputs=False,  
                           **kwargs):
        """skinny wrapper for test_run_vali_multi using dsc formatted results
        
        pars
        --------
        fp_lib: dict
            sim_name
                gridk:filepath
        
        
        """
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rvali', ext='.pkl', **kwargs)
        #if aoi_fp is None: aoi_fp=self.aoi_fp
        
        #=======================================================================
        # precheck
        #=======================================================================        
        assert set(fp_lib.keys()).difference(self.nicknames_d.keys())==set(['inputs'])
        
        
        #detect the inundation file type
        if is_raster_file(inun_fp):
            inun_dkey='INUN_RLAY'
        else:
            inun_dkey='INUN_POLY'
            
        HydTypes(inun_dkey).assert_fp(inun_fp)
        
        #=======================================================================
        # PREP------
        #=======================================================================

        
        """
        print(dstr(fp_lib))
        """

        
        #=======================================================================
        # separate and type check
        #=======================================================================        
        ins_d = fp_lib.pop('inputs')
        dem_fp = ins_d['DEM']
        assert_type_fp(dem_fp, 'DEM')
        
        #rename and check WSE
        log.debug(f'prepping WSE rasters')
        fp_lib2= {k:dict() for k in fp_lib.keys()}
        for sim_name, d in fp_lib.copy().items():
            assert len(d)==1
            vlast = None
            for k,v in d.items():
                #checks
                assert k=='WSE1'
                HydTypes('WSE').assert_fp(v)
                
                if not vlast is None:
                    assert_extent_equal(vlast, v, msg=f'{v} extent mismatch')
                vlast=v
                
                #store
                fp_lib2[sim_name]['WSE'] = v
                
                
        #check inundation
        if inun_dkey=='INUN_RLAY':
            assert_extent_equal(vlast, inun_fp, msg=f'{v} extent mismatch')
                
        
 
        #=======================================================================
        # clip
        #=======================================================================
        """even though it would be nicer to only build WSH on the clip
            easier to keep it on the sub func
            
        force clippiing earlier"""
 
        #=======================================================================
        # get WSH
        #=======================================================================
        log.debug(f'building WSH rasters from {os.path.basename(dem_fp)}')
        """dry filtered"""        
        for sim_name, fp_d in fp_lib2.items():
 
            wse_fp = fp_d['WSE']
            fnm = os.path.splitext(os.path.basename(wse_fp))[0]
            odi = os.path.join(out_dir, sim_name)
            if not os.path.exists(odi):os.makedirs(odi)
            fp_d['WSH'] = get_wsh_rlay(dem_fp, wse_fp, 
                             ofp=os.path.join(odi, f'{fnm}_WSH.tif'),
                             )
            
        #=======================================================================
        # RUN----------
        #======================================================================= 
        log.info(f'computing validation on {len(fp_lib)} sims')
 
        res_lib=dict()
        for sim_name, fp_d in fp_lib2.items():
            #===================================================================
            # setup this observation
            #===================================================================            
            logi = log.getChild(sim_name)
            wse_fp, wsh_fp = fp_d['WSE'], fp_d['WSH']
            logi.info(f'on {sim_name}: WSH={os.path.basename(wsh_fp)}\n\n')
            rdi = dict()
            
            odi = os.path.join(out_dir, sim_name)
            if not os.path.exists(odi):os.makedirs(odi)
            
            skwargs = dict(logger=logi, resname=sim_name, out_dir=odi, tmp_dir=tmp_dir, subdir=False)
            
            
            #=======================================================================
            # HWMs--------
            #======================================================================= 
                
            metric, fp, meta  = self.run_vali_hwm(wsh_fp, hwm_pts_fp, **skwargs)
            rdi['hwm']=dict(metric=metric, fp=fp, meta=meta)
            
            #===================================================================
            # inundation-------
            #===================================================================
            #get the observed inundation (transform)
            inun_rlay_fp = self._get_inun_rlay(inun_fp, wse_fp, **skwargs)
            
            #convert the wsh to binary inundation (0=wet)
            pred_inun_fp = write_extract_mask(wse_fp, invert=True, out_dir=odi)
            assert_spatial_equal(inun_rlay_fp, pred_inun_fp, msg=sim_name)
            
            #run the validation                
            metric, fp, meta = self.run_vali_inun(
                true_inun_fp=inun_rlay_fp, pred_inun_fp=pred_inun_fp, **skwargs) 
            
            rdi['inun']=dict(metric=metric, fp=fp, meta=meta)
            
            
            #===================================================================
            # add inputs
            #===================================================================                
            rdi['clip'] = {'fp':{**fp_d, **{'inun':inun_rlay_fp, 'DEM':dem_fp}}}
            rdi['raw'] = rdi['clip'].copy() #matching fperf format
                           
            #===================================================================
            # wrap
            #===================================================================
            assert len(rdi)>0
            log.debug(f'finished on {sim_name} w/ {len(rdi)}')
            #print(dstr(rdi))
            res_lib[sim_name] = rdi
            
        

        
        #=======================================================================
        # POST---------
        #=======================================================================        
        for k0, v0 in res_lib.items():
            log.debug(f'{k0}-------------------\n'+dstr(v0['raw']['fp']))
 
        #=======================================================================
        # add relative paths
        #=======================================================================
        if self.relative:
            self._add_rel_fp(res_lib) 
      
 
        #print(dstr(res_lib))
        #=======================================================================
        # write meta and pick
        #=======================================================================        
        if write_meta:
            self._write_meta_vali(res_lib)
 
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'finished on \n    {res_lib.keys()}')
        return res_lib
            
        
        
        
        
        
        
                
 
                
            
                             
 
        
 
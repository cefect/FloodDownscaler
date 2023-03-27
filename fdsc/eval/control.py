'''
Created on Mar. 27, 2023

@author: cefect

running evaluation on downscaling results
'''
import os
from hp.basic import dstr
from fdsc.base import DscBaseSession, assert_dsc_res_lib
from fperf.pipeline import ValidateSession

class Dsc_Eval_Session(DscBaseSession, ValidateSession):
    
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
                
 
                
            
                             
    
    def run_dsc_vali_multi(self,
                           dsc_res_lib,
                           **kwargs):
        """build validation on downsample results
        
        Pars
        ---------
        
        """
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rdvX',  **kwargs)
        
 
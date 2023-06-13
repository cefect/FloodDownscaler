'''
Created on Mar. 27, 2023

@author: cefect

running evaluation on downscaling results
'''
import os, pickle, shutil

import rasterio.shutil
from rasterio.enums import Resampling

from hp.basic import dstr
from hp.fiona import get_bbox_and_crs
from hp.hyd import get_wsh_rlay, HydTypes
from hp.rio import (
    write_clip, is_raster_file, copyr, assert_extent_equal, assert_spatial_equal,
    write_resample,
    )
from hp.riom import write_extract_mask

from fdsc.base import assert_dsc_res_lib, DscBaseWorker, assert_type_fp
from fdsc.control import Dsc_Session_skinny
from fperf.pipeline import ValidateSession
 

class Dsc_Eval_Session(ValidateSession, Dsc_Session_skinny):
    """special methods for validationg downscample inundation rasters"""
    
    def _get_fps_from_dsc_lib(self,
                             dsc_res_lib,
                             level=1,
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
        assert_dsc_res_lib(dsc_res_lib, level=level)
        
        log.info(f'on {len(dsc_res_lib)} w/ relative={relative}')
        
        #print(dstr(dsc_res_lib))
        #=======================================================================
        # extract
        #=======================================================================\
        res_d = dict()
        
        #worker
        def get_fpd(d0):
            if relative:
                fp_d = d0['fp_rel']
                return {k:os.path.join(base_dir, v) for k,v in fp_d.items() if not '_raw' in k}
            else:
                return d0['fp']
                
        #loop and extract by level 
        for k0, d0 in dsc_res_lib.items():
            if level==1:
                res_d[k0] = get_fpd(d0)
            elif level==2:
                res_d[k0]=dict()
                for k1, d1 in d0.items():
                    res_d[k0][k1] = get_fpd(d1)
                    
            else:
                raise KeyError(level)
                

                
        

        #=======================================================================
        # check
        #=======================================================================
        def checkit(d0, simName):
            for k, v in d0.items():
                assert os.path.exists(v), f'bad file on {simName}.{k}\n    {v}'
                
        for simName, d0 in res_d.items():
            if level==1:
                checkit(d0, simName)
            elif level==2:
                for k1, d1 in d0.items():
                    checkit(d1, simName)

                
        log.debug('\n'+dstr(res_d))
        
        return res_d
    
    def _add_fps_to_dsc_lib(self, dsc_res_lib, fp_lib, gkey, level=2, **kwargs):
        """add some filepaths"""
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('afps',  **kwargs)
        
        
        #loop and ad by level 
        cnt=0
        for k0, d0 in dsc_res_lib.items():
            if k0=='inputs':
                #d0['inputs1']['fp']
                continue
            if level==1:
                raise IOError('not implemented')
                #res_d[k0] = get_fpd(d0)
            elif level==2:
                
                for k1, d1 in d0.items():
                    
                    
                    
                    #print(fp_lib[k0][k1])
                    assert not gkey in d1['fp']
                    d1['fp'].update({gkey:fp_lib[k0][k1]})
                    
                    #print(k0, k1, '\n', dstr(d1['fp']))
                    cnt+=1
 
                    
            else:
                raise KeyError(level)
            
        log.info(f'updated {cnt} filepaths')
        
        return dsc_res_lib
        
 
    



    def run_vali_multi_dsc(self,
                           fp_lib,
                           #aoi_fp=None,
                           hwm_pts_fp=None,
                           inun_fp=None,
                           write_meta=True,
  
                           **kwargs):
        """skinny wrapper for test_run_vali_multi using dsc formatted results
        
        pars
        --------
        fp_lib: dict
            sim_name
                gridk:filepath
                
        Returns
        -----
        dict (seee load_run_serx() for conversion to multi-index series)
            simName
                analysis (hwm, inun, clip, raw) (clip and raw are the same...)
                    dataType (metric, fp, meta)
                        varName
        
        
        """
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rvali', ext='.pkl', **kwargs)
        #if aoi_fp is None: aoi_fp=self.aoi_fp
        
        #=======================================================================
        # precheck
        #=======================================================================        
        assert set(fp_lib.keys()).difference(self.nicknames_d.values())==set()
        
        
        #detect the inundation file type
        if is_raster_file(inun_fp):
            inun_dkey='INUN_RLAY'
        else:
            inun_dkey='INUN_POLY'
            
        HydTypes(inun_dkey).assert_fp(inun_fp)
        
        #=======================================================================
        # PREP------
        #=======================================================================
 
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
            # inundation (0=wet)-------
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
            rdi['clip'] = {'fp':{**fp_d, **{inun_dkey:inun_fp, 'DEM':dem_fp}}}
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
    
    def run_vali_multi_dsc_aoi(self,
                           fp_lib,
                           aoi_fp=None,
                           hwm_pts_fp=None,
                           inun_fp=None,
                           write_meta=True,
  
                           **kwargs):
        """skinny wrapper for test_run_vali_multi using dsc formatted results
            using a focus aoi (and no building)
        
        pars
        --------
        fp_lib: dict
            outputs from run_vali_multi_dsc() subset to just the inputs (see get_fps_dsc_vali_res_lib())
            simName
                gridKey: filepath
                    
                
                
        Returns
        -----
        dict (seee load_run_serx() for conversion to multi-index series)
            simName
                analType (hwm, inun, clip, raw) (clip and raw are the same...)
                    dataType (metric, fp, meta)
                        varName
        
        
        """
        #print(dstr(fp_lib))
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rvaliA', ext='.pkl', **kwargs)
        #if aoi_fp is None: aoi_fp=self.aoi_fp
 

        #=======================================================================
        # precheck
        #=======================================================================        
        assert set(fp_lib.keys()).difference(self.nicknames_d.values())==set()
        
        
        #detect the inundation file type
        if is_raster_file(inun_fp):
            inun_dkey='INUN_RLAY'
        else:
            inun_dkey='INUN_POLY'
            
        HydTypes(inun_dkey).assert_fp(inun_fp)
        
        #=======================================================================
        # PREP------
        #=======================================================================
        #build container skeleton
        res_lib={simName:{analType:dict() for analType in ['clip']} for simName in fp_lib.keys()}
        
        #add raw
        for simName, d0 in res_lib.items():
            d0['raw']={'fp':fp_lib[simName]}
        #print(dstr(res_lib))
        #=======================================================================
        # clip
        #=======================================================================
        """promote this to a separate step?"""
  
        log.info(f'clipping {len(fp_lib)} w/ {os.path.basename(aoi_fp)}')
        
        bbox, crs = get_bbox_and_crs(aoi_fp)  #set spatial bounds from this
        
        def get_ofp(simName, gkey):
            odi = os.path.join(out_dir, simName)
            if not os.path.exists(odi):os.makedirs(odi)
            return os.path.join(odi, f'{resname}_{simName}_{gkey}.tif')
 
        
        fp_lib2=dict() 
        
        for simName, fp_d in fp_lib.items():
            meta_d, rfp_d=dict(), dict()
 
            for gkey, fp in fp_d.items():
                HydTypes(gkey).assert_fp(fp, msg=simName)
                
                #clip
                ofp, stats_d = write_clip(fp, ofp=get_ofp(simName, f'{gkey}_clip'), 
                                          bbox=bbox, crs=crs, 
                                          fancy_window=dict(round_offsets=True, round_lengths=True),
                                          )
                
                meta_d[gkey]=stats_d
                rfp_d[gkey]=ofp
                
            #store
            fp_lib2[simName]=rfp_d.copy()
            res_lib[simName]['clip']['fp']=rfp_d.copy()
            res_lib[simName]['clip']['meta']=meta_d.copy()
        
        #print(dstr(res_lib))
            
 
        #=======================================================================
        # RUN----------
        #======================================================================= 
        log.info(f'computing validation on {len(fp_lib)} sims')
 
 
 
        for simName, fp_d in fp_lib2.items():
            #print(f'{simName}-------------\n\n{dstr(fp_d)}')
            #===================================================================
            # setup this observation
            #===================================================================            
            logi = log.getChild(simName)
            wse_fp, wsh_fp = fp_d['WSE'], fp_d['WSH']
            logi.info(f'on {simName}: WSH={os.path.basename(wsh_fp)}\n\n')
            rdi = dict()
            
            odi = os.path.join(out_dir, simName)
            if not os.path.exists(odi):os.makedirs(odi)
            
            skwargs = dict(logger=logi, resname=simName, out_dir=odi, tmp_dir=tmp_dir, subdir=False)
            
            
            #=======================================================================
            # HWMs--------
            #======================================================================= 
            """skipping this because there are no obsv within the aoi
            metric, fp, meta  = self.run_vali_hwm(wsh_fp, hwm_pts_fp, **skwargs)
            rdi['hwm']=dict(metric=metric, fp=fp, meta=meta)
            """
            
            #===================================================================
            # inundation (0=wet)-------
            #===================================================================
            """
            print(dstr(fp_d))
            """
            #get the observed inundation (transform)
            inun_rlay_fp = self._get_inun_rlay(inun_fp, wse_fp, **skwargs)
            
            #convert the wsh to binary inundation (0=wet)
            pred_inun_fp = fp_d['INUN_RLAY']
            assert_spatial_equal(inun_rlay_fp, pred_inun_fp, msg=simName)
            
            #run the validation                
            metric, fp, meta = self.run_vali_inun(
                true_inun_fp=inun_rlay_fp, pred_inun_fp=pred_inun_fp, **skwargs) 
            
            rdi['inun']=dict(metric=metric, fp=fp, meta=meta)
            
 
                           
            #===================================================================
            # wrap
            #===================================================================
            assert len(rdi)>0
            log.debug(f'finished on {simName} w/ {len(rdi)}')
            #print(dstr(rdi))
            #print(res_lib[simName].keys())
            res_lib[simName].update(rdi)
            
 
        #=======================================================================
        # POST---------
        #=======================================================================
        #log inputs        
        #for k0, v0 in res_lib.items():
            #log.debug(f'{k0}-------------------\n'+dstr(v0['raw']['fp']))
            #print(f'{k0}-------------------\n'+dstr(v0))
 
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
    
    def build_wsh(self, fp_lib, level=2, **kwargs):
        
        """build corresponding depths"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('bWSH',  **kwargs)
        
        res_d=dict()
        cnt=0
        log.info(f'building WSH grids on {len(fp_lib)}')
        
        assert set(fp_lib.keys()).difference(self.nicknames_d.keys())==set(['inputs'])
        
 
        #=======================================================================
        # loop and build for each
        #=======================================================================
        for k0, d0 in fp_lib.items():
            res_d[k0]=dict()
            #===================================================================
            # handle raw course (and get inputs)
            #===================================================================
            if k0=='inputs':
                ins_d = d0['inputs1'].copy()
                
                #===============================================================
                # #build the course DEM
                # scale = self.get_resolution_ratio(ins_d['WSE2'], ins_d['DEM1'])
                # dem2_fp = write_resample(ins_d['DEM1'], resampling=Resampling.average, 
                #     scale=1/scale, ofp=os.path.join(tmp_dir, f'DEM2_{scale}.tif'))
                # 
                # #get paths
                # odi= os.path.join(out_dir, k0) #matching that of run_dsc_multi_mRes
                # if not os.path.exists(odi):os.makedirs(odi)
                # 
                # wsh2_fp=get_wsh_rlay(
                #         dem2_fp, ins_d['WSE2'], 
                #         ofp=os.path.join(odi, f'WSH_{k0}_{scale}.tif'),
                #         )
                # 
                # print(dstr(ins_d))
                #===============================================================
                continue
            
            if level==1:
                raise IOError('not implemented')
 
            #===================================================================
            # build WSH for each downscale
            #===================================================================
            elif level==2:
                #add the raw
                #res_d[k0]['d00']=wsh2_fp
                
                
                for k1, d1 in d0.items():
                    #get paths
                    odi= os.path.join(out_dir, k0, k1) #matching that of run_dsc_multi_mRes
                    if not os.path.exists(odi):os.makedirs(odi)
                    
                    #build from DEM
                    res_d[k0][k1]=get_wsh_rlay(
                        ins_d[k1], d1['WSE1'], 
                        ofp=os.path.join(odi, f'WSH_{k0}{k1}.tif'),
                        )
                        
                    cnt+=1
                    

                      
        #=======================================================================
        # wrap  
        #=======================================================================
        log.info(f'finished writing {cnt} to \n    {out_dir}')
        """
        print(dstr(res_d))
        """
        
        return res_d
 
                    
    
    def run_stats_multiRes(self,
                           fp_lib,
                           level=2,
  
                           **kwargs):
        """compute grid stats
        
        pars
        --------
        fp_lib: dict
            sim_name
                downscale_str
                    gridk:filepath
        
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('rstats', ext='.pkl', **kwargs)
 

        
        res_lib=dict()
        cnt=0
        log.info(f'building WSH grids on {len(fp_lib)}')
        
        #=======================================================================
        # precheck
        #=======================================================================        
        assert set(fp_lib.keys()).difference(self.nicknames_d.keys())==set(['inputs'])
        
        #=======================================================================
        # loop and build for each
        #=======================================================================
        for k0, d0 in fp_lib.items():
            if k0=='inputs':
                ins_d = d0.copy()
                continue
            
            if level==1:
                raise IOError('not implemented')
 
            elif level==2:
                res_lib[k0]=dict()
                for k1, d1 in d0.items():
                    res_lib[k0][k1]=dict()
 
                    
                    for gridk, fp in d1.items():
                        if gridk=='WSH1':
                            #extract
                            with HydTypes('WSH', fp=fp) as wrkr:
                                res_lib[k0][k1][gridk] = wrkr.WSH_stats()
                                
                            
                            cnt+=1
 
 
        #=======================================================================
        # wrap  
        #=======================================================================
        log.info(f'finished computing on {cnt} grids')
        
        return res_lib
            
        
        
        
        
        
        
                
 
                
            
                             
 
        
 
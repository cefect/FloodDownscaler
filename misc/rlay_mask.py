'''
Created on Dec. 23, 2020

@author: cefect

clip a set of raster layers by an aoi layer
'''
#===============================================================================
# script paths
#===============================================================================
work_dir = r'C:\LS\03_TOOLS\LML\kent'
out_dir = r'C:\LS\03_TOOLS\LML\_outs\kent'
mod_name = 'rlay_mask'
#===============================================================================
# # standard imports -----------------------------------------------------------
#===============================================================================
import time, sys, os, logging, copy, shutil, re, inspect, weakref, fnmatch

import numpy as np
import pandas as pd

from qgis.core import QgsCoordinateReferenceSystem, QgsVectorLayer, QgsMapLayerStore, \
    QgsWkbTypes

#===============================================================================
# logging
#===============================================================================
mod_logger = logging.getLogger(__name__)

#===============================================================================
# custom imports
#===============================================================================

from hp.Q import Qproj, view
from hp.exceptions import Error
from hp.dirz import force_open_dir


class Sliceor(Qproj):
    
    aoi_vlay = None
        
    def __init__(self,
                 aoi_fp = None,
                 outResolution = 10, #resolution for output raster
                 crsOut = None,
                 **kwargs):
        
        super().__init__(mod_name=mod_name, **kwargs)  # initilzie teh baseclass
        
        
        #=======================================================================
        # attach
        #=======================================================================
        self.outResolution=outResolution
        self.crsOut=crsOut
        #=======================================================================
        # setups
        #=======================================================================
        if not aoi_fp is None:
            #get the aoi
            aoi_vlay = self.load_aoi(aoi_fp)
            
        
        
    def run_clip(self, #load a set of layers, slice by aoi, report on feature counts
                  data_dir,
                  fname_sfx = 'aoi02',
                  logger=None,
                  ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('run_slice')
        
        

        
        #collect filepaths
        mdf = self.get_layFp_dir(data_dir, logger=log)
        
        #=======================================================================
        # #load, clip, save
        #=======================================================================
        #setup the filename pars based on the containing folder
        mdf = mdf.rename(columns={'folderName':'fname'})
        
        mdf.loc[:, 'fname'] = mdf['fname'] + '_%s.tif'%fname_sfx
        
        
        
        
        
        
        self.meta_d = self.slice_to_file(mdf.to_dict(orient='index'))
        
        return self.meta_d
 

    def get_layFp_dir(self, #collect filepaths to layers from a directory
                        data_dir,  # top level directory to search
                        
                        search_patterns=('*.tif',),  # set of Unix shell-style wildcards
                        
                        logger=None,
                        ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('load_rlays_dir')
        assert os.path.exists(data_dir)
 
        #=======================================================================
        # walk and load
        #=======================================================================
        #containers

        mdf = None #metadata container
        
        
        for dirpath, dirnames, filenames in os.walk(data_dir):
            log.debug('%s    %s    %s' % (dirpath, dirnames, filenames))
            
            # get the name of the current folder
            _, folderName = os.path.split(dirpath)
 
            log.info('loading from \'%s\' w/ %i files \n    %s' % (
                folderName, len(filenames), filenames))
            
            # load teh layers here
            #first = True
            for fileName in filenames:
                baseFileName, ext = os.path.splitext(fileName)
                
                #===============================================================
                # #check if this file matches the search patterns
                #===============================================================
                match_l = list()
                for searchPattern in search_patterns:
                    if not fnmatch.fnmatch(fileName, searchPattern):
                        match_l.append(searchPattern)
                        
                if len(match_l)>0:
                    log.debug('\'%s\' failed %i searches...skipping:   %s'%(
                        fileName, len(match_l), match_l))
                    continue #

 
                    
                #===============================================================
                # load and clip the file
                #===============================================================
                #get the raw filepath
                fp = os.path.join(dirpath, fileName)
                

                
                #===============================================================
                # #meta
                #===============================================================
                mdfi = pd.Series({
                    'folderName':folderName,
                    'fp':fp,
                    'baseFileName':baseFileName},
                        name = str('%s.%s'%(folderName, baseFileName[:15])).replace(' ','')
                        ).to_frame().T
                
                if mdf is None:
                    mdf = mdfi
                else:
                    mdf = mdf.append(mdfi)
 
                
            #===================================================================
            # wrap page
            #===================================================================
 
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished w/ %i pages' % len(mdf))
        
        return mdf
    
    def slice_to_file(self,
                      fp_lib, #{layerName: {fp:filepath, fname:new filename}}
                      aoi_vlay = None,
                      outResolution=None, 
                      crsOut=None,
                      ):
        
        log = self.logger.getChild('slice_to_file')
        if aoi_vlay is None: aoi_vlay=self.aoi_vlay
        if outResolution is None: outResolution=self.outResolution
        if crsOut is None: crsOut=self.crsOut
        #=======================================================================
        # loop on rlays
        #=======================================================================
        log.info('on %i \n    %s'%(len(fp_lib), list(fp_lib.values())))
        meta_lib = dict()
        for layName, d in fp_lib.items():

            fp = d['fp']
            
            #===================================================================
            # #filename
            #===================================================================
            if 'fname' in d:
                fname = d['fname']
            else:
                bdir, ext = os.path.splitext(fp)
                fname = '%s_%s'%(os.path.basename(bdir), self.aoi_vlay.name())+ext
            
            output = os.path.join(self.out_dir,fname)

            
            #===================================================================
            # clip to file and re-project
            #===================================================================
            _ = self.rlay_load(fp , aoi_vlay=aoi_vlay, logger=log, 
                   output=output, result='fp', outResolution=outResolution,#kwargs for cliprasterwithpolygon
                                   crsOut=crsOut)
            
            
            #meta
            meta_lib[layName] = {
                'og_fp':d['fp'], 'new_fp':output, 'outCrs':crsOut,
                }
            
        log.info('finished on %i'%len(meta_lib))
        
        return meta_lib
    
    def write_meta(self, d=None, logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        if d is None: d=self.meta_d
        if logger is None: logger=self.logger
        log=logger.getChild('write_meta')
        
        df = pd.DataFrame.from_dict(d, orient='index')
        
        ofp = os.path.join(self.out_dir, '%s_%s_%i_meta.csv'%(self.mod_name, self.tag, len(df)))
        
        if os.path.exists(ofp):assert self.overwrite
        df.to_csv(ofp, index=True)
        
        log.info('wrote %s to file \n    %s'%(str(df.shape), ofp))

        return ofp

    
#===============================================================================
# runners----------
#===============================================================================
        
def run_dir(runPars_d, #use directyory searches
        overwrite=True):
    
    
    for tag, pars_d in runPars_d.items():
        
        kwargs = {k:v for k,v in pars_d.items() if k in [
            'aoi_fp', 'crsID_default', 'out_dir', 'crsOut']}
        wrkr = Sliceor(**kwargs, tag=tag, overwrite=overwrite)
        
        kwargs = {k:v for k,v in pars_d.items() if k in ['data_dir']}
        _ = wrkr.run_clip(**kwargs)
        
        wrkr.write_meta()
    
    return wrkr.out_dir

def run_slice(runPars_d, overwrite=True): #thin slice to file wrapper
    for tag, pars_d in runPars_d.items():
        
        #=======================================================================
        # init
        #=======================================================================
        kwargs = {k:v for k,v in pars_d.items() if k in [
            'aoi_fp', 'crsID_default', 'out_dir', 'crsOut']}
        
        wrkr = Sliceor(**kwargs, tag=tag, overwrite=overwrite)
        
        #=======================================================================
        # load and slice
        #=======================================================================
        output = os.path.join(wrkr.out_dir, pars_d['fname'])
        kwargs = {k:v for k,v in pars_d.items() if k in ['outResolution', 'crsOut']}
        ofp = wrkr.rlay_load(pars_d['fp'] , aoi_vlay=wrkr.aoi_vlay, result='fp',output=output,
                              **kwargs)
        
    return wrkr.out_dir
    

if __name__ == '__main__':
    
    runPars_d = {
        #=======================================================================
        # 't1':{
        #     'data_dir':r'C:\LS\03_TOOLS\misc\_in\20210303',
        #     'aoi_fp':r'C:\LS\03_TOOLS\misc\_in\20210303\aoi01.gpkg',
        #     'crsID_default':'EPSG:3005'
        #     },
        #=======================================================================
        #=======================================================================
        # 't2':{
        #     'data_dir':r'C:\LS\03_TOOLS\misc\_in\kent_test',
        #     'aoi_fp':r'C:\LS\02_WORK\NHC\202012_Kent\04_CALC\aoi\Kent_aoi02b_20210303_e26910.gpkg',
        #     'crsID_default':'EPSG:26910',
        #     'crsOut':QgsCoordinateReferenceSystem('EPSG:3005'),
        #     },
        #=======================================================================
        #=======================================================================
        # 'b1':{
        #     'data_dir':r'e:\02_WORK\NHC\202012_Kent\01_GEN\2021 03 03 - Natalia - 12breach runs',
        #     'aoi_fp':r'C:\LS\02_WORK\NHC\202012_Kent\04_CALC\aoi\Kent_aoi02b_20210303_e26910.gpkg',
        #     'crsID_default':'EPSG:26910',
        #     'out_dir':r'C:\LS\03_TOOLS\misc\outs\b1',
        #     'crsOut':QgsCoordinateReferenceSystem('EPSG:3005'),
        #     },
        #=======================================================================
        'dtm':{
            'fp':r'E:\04_LIB\NHC\2019-Fraser\DTM\Terrain_Bridges_CoqSum_Extended.g3003429MosUS_VD2013_CoqSumBrdgExt.tif',
            'crsOut':QgsCoordinateReferenceSystem('EPSG:3005'),'crsID_default':'EPSG:26910',
            'aoi_fp':r'C:\LS\02_WORK\NHC\202012_Kent\04_CALC\aoi\Kent_aoi02b_20210303_e26910.gpkg',
            'outResolution':None, 
            'fname':'NHC2019_DTM_hires_aoi02b.tif',
            }
        
        }
    
    out_dir = run_slice(runPars_d)

    force_open_dir(out_dir)
    print('finished')

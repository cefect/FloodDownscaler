'''
Created on Dec. 23, 2020

@author: cefect

slice a set of layers and report on feature counts
'''
#===============================================================================
# script paths
#===============================================================================
work_dir = r'C:\LS\03_TOOLS\LML\kent'
out_dir = r'C:\LS\03_TOOLS\LML\_outs\kent'

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


class Sliceor(Qproj):
    
    aoi_vlay = None
        
    def __init__(self,
                 aoi_fp = None,
                 **kwargs):
        
        super().__init__(

                         **kwargs)  # initilzie teh baseclass
        
        if not aoi_fp is None:
            #get the aoi
            aoi_vlay = self.load_aoi(aoi_fp)
            
        
        
    def run_slice(self, #load a set of layers, slice by aoi, report on feature counts
                  data_dir,
                  aoi_vlay = None,
                  logger=None,
                  ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('run_slice')
        
        #setup aoi
        """typically done during the init"""
        if not aoi_vlay is None:
            self.aoi_vlay = aoi_vlay
        
        #load the working set
        layRaw_d, mdf = self.load_layers_dir(data_dir, logger=log)
        

        
        #slice each by aoi and get some stats
        laySlice_d, cnt_df = self.slice_aoi_set(layRaw_d, logger=log)
        
        #join in
        mdf = mdf.join(cnt_df)
        
        return laySlice_d, mdf

    def load_layers_dir(self,
                        data_dir,  # top level directory to search
                        
                        search_patterns=('*.gpkg',),  # set of Unix shell-style wildcards
                        ext='.gpkg',
                        logger=None,
                        ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('loadLD')
        assert os.path.exists(data_dir)
        
        log.info('loading w/ %s from %s'%(search_patterns, data_dir))
        
        #=======================================================================
        # walk and load
        #=======================================================================
        #containers
        layers_d = dict()
        mdf = pd.DataFrame(columns = ('folderName', 'fp', 'geometry')) #metadata container
        
        
        for dirpath, dirnames, fns_raw in os.walk(data_dir):
            log = logger.getChild('loadLD.%s'%os.path.basename(dirpath))
            #log.debug('%s    %s    %s' % (dirpath, dirnames, filenames))
            
            # get the name of the current folder
            _, folderName = os.path.split(dirpath)
            assert folderName not in layers_d
            
            #clean out filenames
            filenames = [fn for fn in fns_raw if fn.endswith(ext)]
            
            if len(filenames)==0: 
                log.debug('no relevant files in %s'%dirpath)
                continue
            
            log.info('loading from \'%s\' w/ %i files \n    %s' % (
                folderName, len(filenames), filenames))
            
            # load teh layers here
            #first = True
            for fileName in filenames:
                baseFileName, exti = os.path.splitext(fileName)
                #if not ext == exti: continue #ignore this file
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

                

                assert not baseFileName in layers_d
                    
                # load the file
                fp = os.path.join(dirpath, fileName)
                layer = self.vlay_load(fp , logger=log)
                
                # add it
                log.debug('adding %s.%s' % (folderName, baseFileName))
                layer.setName(baseFileName)
                layers_d[baseFileName] = layer
                
                #meta
                """
                todo: make this a series w/ dict
                """
                mdf.loc[baseFileName, 'folderName'] = folderName
                mdf.loc[baseFileName, 'fp'] = fp
                mdf.loc[baseFileName, 'geometry'] = QgsWkbTypes().displayString(layer.wkbType())
                mdf.loc[baseFileName, 'fn'] = fileName
                
                
            """one page
            if folderName in layers_d:
                layerCnt = len(layers_d[folderName])
                assert layerCnt >=1, 'nothing loaded for %s'%folderName
            else:
                layerCnt = 0"""
            layerCnt = len(layers_d)
                
            #===================================================================
            # wrap page
            #===================================================================
            log.info('finished \'%s\' w/ %i layers' % (folderName, layerCnt))
            
        log = logger.getChild('loadLD')
        assert len(layers_d)>0, 'no layers loaded!'
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished w/ %i pages' % len(layers_d))
        
        return layers_d, mdf
    

    
    
    def slice_aoi_set(self, #slice a set and get some info
                  layers_d,
                  delete_raw = True,
                  logger=None,
                  **slice_aoi_kwargs
                  ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('aoi_slice')
        
        
        
        if delete_raw:
            mstore = QgsMapLayerStore() #build a new map store for removing the old layers
        else:mstore=None
        #=======================================================================
        # prechecks
        #=======================================================================
        assert len(layers_d)>0, 'got no layers'
        
        #=======================================================================
        # get raw counts
        #=======================================================================
        d = self.get_featureCounts(layers_d, logger=log)
        cnt_df = pd.Series(d, name='cnt_raw').to_frame()
        
        #=======================================================================
        # loop and slice
        #=======================================================================
        log.info('slicing %i layers'%(len(layers_d)))
        
        res_d = dict()
        for loopName, vlayRaw in layers_d.items():
            
            
            vlaySlice = self.slice_aoi(vlayRaw,logger=log,
                                       **slice_aoi_kwargs)
            
            # no selection
            if vlaySlice is None:
                fcnt = 0
                
                res_d[loopName] = None
            else:
                fcnt = vlaySlice.dataProvider().featureCount()
                res_d[loopName] = vlaySlice
                cnt_df.loc[loopName, 'fn_slice'] =  vlaySlice.name()
                
            #store
            cnt_df.loc[loopName, 'cnt_aoi'] =  fcnt
            
            
            
            
            #cleanup
            if not mstore is None: 
                mstore.addMapLayer(vlayRaw)
                mstore.removeAllMapLayers()
            
            log.debug('%s w/ %i'%(loopName, cnt_df.loc[loopName, 'cnt_aoi']))
            
            
        #=======================================================================
        # wrap
        #=======================================================================
        #combine results data
        cnt_df['cnt_aoi'] = cnt_df['cnt_aoi'].astype(int)
        

        log.info('finished slicing %i'%len(res_d))
        
        return res_d, cnt_df
            

    def get_featureCounts(self,  # get the feature count of each layer
                          layers_d,
                          logger = None,
                          ):
            
            #===================================================================
            # defaults
            #===================================================================
            if logger is None: logger=self.logger
            log = logger.getChild('get_featureCounts')
            
            log.info('on %i layers' % len(layers_d))
                
            #===============================================================
            # retrieve for each layer
            #===============================================================
            res_d = dict()  # count per layer
            for layName, layer in layers_d.items():
                assert isinstance(layer, QgsVectorLayer), layName
                assert layer.name() == layName
                
                dp = layer.dataProvider()
                res_d[layName] = dp.featureCount()
                
                assert dp.featureCount()>0, layName
                
                
            assert len(res_d)==len(layers_d)
            log.info('got count on %i layers: \n    %s'%(len(res_d), res_d))
                
            return res_d
        
    def save_all_layers(self,
                 layers_d,
                 out_dir = None,
                 logger=None,
                 ignore_blank = True, #whether to allow 'None' entries
                 **kwargs):
        
        if logger is None: logger=self.logger
        log=logger.getChild('save_all_layers')
        if out_dir is None: out_dir=self.out_dir
        
        #=======================================================================
        # prechecks
        #=======================================================================
        assert os.path.exists(out_dir)
        
        log.info("writing %i layers to directory: \n    %s"%(len(layers_d), out_dir))
        #=======================================================================
        # loop and write
        #=======================================================================
        wcnt = 0
        for layTag, layer in layers_d.items():
            #empty check
            if layer is None:
                msg = '%s is empty: ignore_blank=%s'%(layTag, ignore_blank)
                if ignore_blank:
                    log.debug(msg)
                    continue #skipit
                else:
                    raise Error(msg)
            
            #filepath and write
            fp = os.path.join(out_dir, '%s.gpkg'%layer.name())
            self.vlay_write(layer, fp, logger=log, **kwargs)
            wcnt+=1
            
        log.info('finished writing %i/%i to file'%(wcnt, len(layers_d)))
        
        return
        


if __name__ == '__main__':
    
    data_dir = r'C:\LS\03_TOOLS\LML\_ins2\kent'
    aoi_fp = r'C:\LS\02_WORK\NHC\202012_Kent\04_CALC\aoi\BC_MoMAH_Kent_aoi01a.gpkg'
    
    
    wrkr = Sliceor(aoi_fp=aoi_fp)
    
    
    laySlice_d, mdf = wrkr.run_slice(data_dir)
    
    """
    view(mdf)
    """
    
    
    
    print('finished')

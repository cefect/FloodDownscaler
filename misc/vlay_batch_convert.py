'''
Created on Dec. 23, 2020

@author: cefect

batch convert a set of vector files
'''
#===============================================================================
# script paths
#===============================================================================
search_dir = r'C:\LS\02_WORK\02_Mscripts\InsuranceCurves\06_DATA\FloodsInCanada\20210924\extract'
out_dir = r'C:\LS\02_WORK\02_Mscripts\InsuranceCurves\06_DATA\FloodsInCanada\20210924\gpkg'

#===============================================================================
# # standard imports -----------------------------------------------------------
#===============================================================================
import time, sys, os, logging, copy, shutil, re, inspect, weakref, fnmatch

import numpy as np
import pandas as pd

from qgis.core import QgsCoordinateReferenceSystem, QgsVectorLayer, QgsMapLayerStore, \
    QgsWkbTypes



#===============================================================================
# custom imports
#===============================================================================

from hp.Q import Qproj, view
from hp.exceptions import Error


class Convertor(Qproj):
    

        
    def __init__(self,
                 
                 **kwargs):
        
        super().__init__(

                         **kwargs)  # initilzie teh baseclass

            
        
        
    def run_conv(self, #load a set of layers, slice by aoi, report on feature counts
                  data_dir,
 
                  logger=None,
                  ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('run_slice')
        
 
        
        #load the working set
        layRaw_d, mdf = self.load_layers_dir(data_dir, logger=log)
        

        
        #slice each by aoi and get some stats
        lay_d, cnt_df = self.export_set(layRaw_d, logger=log)
        
        #join in
        mdf = mdf.join(cnt_df)
        
        return lay_d, mdf

    def load_layers_dir(self,
                        data_dir,  # top level directory to search
                        
                        search_patterns=('*.shp','FloodExtentPolygon*'),  # set of Unix shell-style wildcards
                        logger=None,
                        ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('load_layers_dir')
        assert os.path.exists(data_dir)
        
        
        
        #=======================================================================
        # walk and load
        #=======================================================================
        #containers
        layers_d = dict()
        mdf = pd.DataFrame(columns = ('folderName', 'fp', 'geometry')) #metadata container
        
        
        for dirpath, dirnames, filenames in os.walk(data_dir):
            log.debug('%s    %s    %s' % (dirpath, dirnames, filenames))
            
            # get the name of the current folder
            _, folderName = os.path.split(dirpath)
            assert folderName not in layers_d
            
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

                
                """just using one page
                #start the page
                if first:
                    layers_d[folderName] = dict()
                    first =False
                    
                    
                if baseFileName in layers_d[folderName]:
                    raise Error('file already there!: %s.%s'%(folderName, baseFileName))
                """
                assert not baseFileName in layers_d
                    
                # load the file
                fp = os.path.join(dirpath, fileName)
                layer = self.vlay_load(fp , logger=log)
                
                # add it
                log.debug('adding %s.%s' % (folderName, baseFileName))
                layer.setName(baseFileName)
                layers_d[baseFileName] = layer
                
                #meta
                mdf.loc[baseFileName, 'folderName'] = folderName
                mdf.loc[baseFileName, 'fp'] = fp
                mdf.loc[baseFileName, 'geometry'] = QgsWkbTypes().displayString(layer.wkbType())
                
                
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
            
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finished w/ %i pages' % len(layers_d))
        
        return layers_d, mdf
    

    
    def export_set(self, #slice a set and get some info
                  layers_d,
                  delete_raw = False,
                  logger=None,
                  **slice_aoi_kwargs
                  ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log = logger.getChild('export_set')
        

        
        if delete_raw:
            mstore = QgsMapLayerStore() #build a new map store for removing the old layers
        else:mstore=None
        #=======================================================================
        # prechecks
        #=======================================================================
        
        
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
            

                
            #meta
            fcnt = vlayRaw.dataProvider().featureCount()
            cnt_df.loc[loopName, 'cnt_aoi'] =  fcnt
            
            #export
            self.vlay_write(vlayRaw, os.path.join(self.out_dir, loopName+'.gpkg'), logger=log)
            
            
            
            #cleanup
            if not mstore is None: mstore.addMapLayer(vlayRaw)
            
            log.debug('%s w/ %i'%(loopName, cnt_df.loc[loopName, 'cnt_aoi']))
            
            
        #=======================================================================
        # wrap
        #=======================================================================
        #combine results data
        cnt_df['cnt_aoi'] = cnt_df['cnt_aoi'].astype(int)
        
        #memory clear
        if not mstore is None:
            mstore.removeAllMapLayers()
            log.warning('cleared %i raw layers from memory'%len(layers_d))
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
        


        


if __name__ == '__main__':
    
 
    
    
    wrkr = Convertor(out_dir=out_dir)
    
    
    laySlice_d, mdf = wrkr.run_conv(search_dir)
    
    """
    view(mdf)
    """
    
    
    
    print('finished')

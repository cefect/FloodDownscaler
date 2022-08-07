'''
Created on Aug. 7, 2022

@author: cefect
'''
import os
import numpy as np
 
import numpy.ma as ma
import rasterio
import rasterio.features
import rasterio.warp
 
from rasterio.enums import Resampling

from hp.oop import Basic
from hp.plot import plot_rast #for debugging


class RioWrkr(Basic):
    """work session for single band raster calcs"""
    
    driver='GTiff'
    bandCount=1
    
    def __init__(self,
                 rlay_ref_fp = None,  
                 
                 #reference inheritance
                 crs=None,dtype=None,height=None,width=None,transform=None,
                 **kwargs):
        """"
        
        Parameters
        -----------
        """
 
        super().__init__(**kwargs)
        
        self.dataset_d = dict() #all loaded datasets
        
        

            
 
 
        #=======================================================================
        # set reference
        #=======================================================================        
        if not rlay_ref_fp is None:
            rds = self.open_dataset(rlay_ref_fp, meta=False)
            self.ref_name = rds.name
            #data=rds.read()
        else:
            rds=None
            self.ref_name=None
            #data=None
            
           
        #=======================================================================
        # inherit properties from reference 
        #=======================================================================
        pars_d = dict()
        def inherit(attVal, attName, obj=rds, typeCheck=None):
            if attVal is None:
                assert not obj is None, 'for \'%s\' passed None but got no rds'%attName
                attVal = getattr(obj, attName)
            assert not attVal is None, attName
            setattr(self, attName, attVal)
            
            pars_d[attName] = attVal
 
                    
            if not typeCheck is None:
                assert isinstance(getattr(self, attName), typeCheck), \
                    'bad type on \'%s\': %s'%(attName, type(getattr(self, attName)))
            
        #retrieve defaults
        inherit(crs,  'crs')
        #inherit(dtype,  'dtype', obj=data)
        inherit(height,  'height')
        inherit(width,  'width')
        inherit(transform, 'transform')
        
        self.logger.info('init w/ %s'%pars_d)
        
 
            
    
    def open_dataset(self,
                     fp,
                     logger=None,
                     meta=True,
                     **kwargs):
        """open a dataset"""
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('open_ds')
        assert os.path.exists(fp), fp
        log.debug('open: %s'%fp)
        dataset = rasterio.open(fp, mode='r', **kwargs)
        
        #=======================================================================
        # #meta
        #=======================================================================
        if meta:
            dataset.profile
            assert dataset.count==self.bandCount, 'only setup for single band'
            msk = dataset.read_masks(self.bandCount)  #read the GDAL RFC 15 mask
            nodata_cnt = (msk==0).sum()
     
        
            d = {'name':dataset.name, 'shape':str(dataset.shape),
             'nodata_cnt':nodata_cnt, 'size':np.prod(dataset.shape),
             'crs':dataset.crs, 'ndval':dataset.nodata}
            log.info('loaded {shape} raster \'{name}\' on {crs} w/  {nodata_cnt}/{size} nulls (ndval={ndval}) '.format(**d))
        
        #=======================================================================
        # #attach
        #=======================================================================
        self.dataset_d[dataset.name] = dataset
        
        return dataset
        
    
        
    def __enter__(self):
        return self
    
    def __exit__(self,  *args,**kwargs):
 
        #close all open datasets
        for k,v in self.dataset_d.items():
            v.close()
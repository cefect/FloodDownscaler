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
                 crs=None,height=None,width=None,transform=None,nodata=None,
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
        inherit(nodata, 'nodata')
        
        self.ref_vals_d=pars_d
        
        self.logger.info('init w/ %s'%pars_d)
        
 
            
    

    
    
    
    def resample(self,
 
                 resampling=Resampling.nearest,
                 scale=1.0,
                 write=True,
                 update_ref=False, 
                 **kwargs):
        """"
        Parameters
        ---------
        
        update_ref : bool, default False
            Whether to update the reference values based on the resample
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger, log, dataset, out_dir, ofp = self._func_kwargs(name = 'resample_r%i'%scale, **kwargs)
        
 
        
        log.info('on %s w/ %s'%(dataset.name, dict(resampling=resampling, scale=scale)))
        
        #===========================================================================
        # # resample data to target shape
        #===========================================================================
        out_shape=(dataset.count,int(dataset.height * scale),int(dataset.width * scale))
        print('transforming from %s to %s'%(dataset.shape, out_shape))
        data_rsmp = dataset.read(
            out_shape=out_shape,
            resampling=resampling
                                )[0]
        
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data_rsmp.shape[-1]),
            (dataset.height / data_rsmp.shape[-2])
        )
        
        
        #===========================================================================
        # resample nulls
        #===========================================================================
        msk_rsmp = dataset.read_masks(1, 
                out_shape=out_shape,
                resampling=Resampling.nearest, #doesnt bleed
            ) 
        
        #===============================================================================
        # coerce transformed nulls
        #===============================================================================
        """needed as some resampling methods bleed out
        theres a few ways to handle this... 
            here we manipulate the data values directly.. which seems the cleanest"""
 
        assert data_rsmp.shape==msk_rsmp.shape
        data_rsmp_f1 = np.where(msk_rsmp==0,  dataset.nodata, data_rsmp)
        
        log.info('resampled from %s to %s'%(dataset.shape, data_rsmp.shape))
        
        #=======================================================================
        # write
        #=======================================================================
        if not write:
            res= data_rsmp_f1
        else:
            res =self.write_dataset(data_rsmp_f1, ofp=ofp, logger=log, transform=transform)
        
        #=======================================================================
        # update
        #=======================================================================
        if update_ref:
            raise IOError('not implemented')
        
        return res
    
    #===========================================================================
    # HELPERS----------
    #===========================================================================
    
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
        dataset = rasterio.open(fp, mode='r', name='test', **kwargs)
        
        #=======================================================================
        # clean up the name
        #=======================================================================
        try:
            dataset.clean_name=os.path.basename(dataset.name).replace('.tif', '')
        except Exception as e:
            log.warning('failed to build clean_name w/ %s'%e)
            dataset.clean_name = dataset.name
        
        
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
    
    
    def get_ndcnt(self, **kwargs):
        logger, dataset, *args = self._func_kwargs(**kwargs)
        
        msk = dataset.read_masks(1)  #read the GDAL RFC 15 mask
        nodata_cnt = (msk==0).sum()
        
        del msk
        return nodata_cnt
 
    def write_dataset(self,data,
                       
                       crs=None,nodata=None,transform=None,
                       **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger, log, _, out_dir, ofp = self._func_kwargs(name = 'write', **kwargs)
        
        crs, _, _, transform, nodata = self._get_refs(crs=crs, nodata=nodata, transform=transform)
        
        #=======================================================================
        # write
        #=======================================================================
        with rasterio.open(
                ofp,
                'w',
                driver=self.driver,
                height=data.shape[0],
                width=data.shape[1],
                count=self.bandCount,
                dtype=data.dtype,
                crs=crs,
                transform=transform,
                nodata=nodata,
                ) as dst:
            
                dst.write(data, 1, 
                          masked=False, #not using numpy.ma
                          )
                
        log.info('wrote %s on crs %s to \n    %s'%(str(data.shape), crs, ofp))
        
        return ofp
        
    #===========================================================================
    # PRIVATES---------
    #===========================================================================
    def _func_kwargs(self, logger=None, dataset=None, out_dir=None, ofp=None,name=None):
        """typical default for class functions"""
 
        if logger is None:
            logger=self.logger
 
        
        if not name is None:
            log = logger.getChild(name)
        else:
            log = logger
 
        
        if dataset is None:
            dataset = self.dataset_d[self.ref_name]
        
 
        if out_dir is None:
            out_dir=self.out_dir
            
        if ofp is None:
            if name is None:
                ofp = os.path.join(out_dir, self.fancy_name + '.tif')
            else:
                ofp = os.path.join(out_dir, self.fancy_name + '_%s.tif'%name)
            
            
        return logger, log, dataset, out_dir, ofp
    
    def _get_refs(self, **kwargs):
        
        def get_aval(attName):
            if attName in kwargs:
                attVal=kwargs[attName]
            else:
                attVal = None
                
            if attVal is None:
                attVal = getattr(self, attName)
                
            return attVal
            
        args=list()
        for attName in ['crs', 'height', 'width', 'transform', 'nodata']:
            args.append(get_aval(attName))
        
        #self.ref_vals_d.keys()
        
        return args #crs, height, width, transform, nodata
        
            
        
        
    
        
    def __enter__(self):
        return self
    
    def __exit__(self,  *args,**kwargs):
 
        #close all open datasets
        for k,v in self.dataset_d.items():
            v.close()
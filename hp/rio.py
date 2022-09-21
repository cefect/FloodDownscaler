'''
Created on Aug. 7, 2022

@author: cefect
'''
import os
import numpy as np
 
import numpy.ma as ma
import rasterio as rio
import shapely.geometry as sgeo
 
#print('rasterio.__version__:%s'%rio.__version__)
 
assert os.getenv('PROJ_LIB') is None, 'rasterio expects no PROJ_LIB but got \n%s'%os.getenv('PROJ_LIB')
 
import rasterio.merge
import rasterio.io
from rasterio.plot import show
from rasterio.enums import Resampling

import scipy.ndimage
#import skimage
#from skimage.transform import downscale_local_mean

#import hp.gdal
from hp.oop import Basic
from hp.basic import get_dict_str
#from hp.plot import plot_rast #for debugging
import matplotlib.pyplot as plt

class RioWrkr(Basic):
    """work session for single band raster calcs"""
    
    driver='GTiff'
    bandCount=1
    ref_name=None
    
    #base attributes
    """nice to have these on the class for more consistent retrival... evven if empty"""
    nodata=None
    crs=None
    height=None
    width=None
    transform=None
 


    def __init__(self,
                 rlay_ref_fp = None,  
                 
                 #default behaviors
                 compress=None,
                 
                 #reference inheritance
                 crs=None,height=None,width=None,transform=None,nodata=None,
                 
                 **kwargs):
        """"
        
        Parameters
        -----------
        """
 
        super().__init__(**kwargs)
        
        self.dataset_d = dict() #all loaded datasets
        self.memoryfile_d=dict()
        
        #=======================================================================
        # simple attachments
        #=======================================================================
        self.compress=compress
 
        #=======================================================================
        # set reference
        #=======================================================================        
        if not rlay_ref_fp is None:
            self._base_set(rlay_ref_fp)
            
        #=======================================================================
        # inherit properties from reference 
        #=======================================================================
        pars_d=self._base_inherit(crs=crs, height=height, width=width, transform=transform, nodata=nodata)
        
        self.logger.debug('init w/ %s'%pars_d)
        
    def _base_set(self, rlay_ref_fp, **kwargs):
        rds = self.open_dataset(rlay_ref_fp, meta=False, **kwargs)
        self.ref_name = rds.name
        return rds
        
    def _base_inherit(self,
                      ds=None,
                      crs=None, height=None, width=None, transform=None, nodata=None):
        
        #retrieve the base datasource
        if ds is None:
            ds = self._base()
        
        
        pars_d = dict()
        def inherit(attVal, attName, obj=ds, typeCheck=None):
            if attVal is None:
                if obj is None:
                    self.logger.debug('no value passed for %s'%attName)
                    return
                    
                attVal = getattr(obj, attName)
            #assert not attVal is None, attName
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
        
        return pars_d            
    

    
    #===========================================================================
    # MANIPULATORS----------
    #===========================================================================
    
    def resample(self,
 
                 resampling=Resampling.nearest,
                 scale=1.0,
                 #write=True,
                 #update_ref=False, 
                 prec=None,
                 name=None,
                 **kwargs):
        """"resample a rio.dataset handling nulls
        
        
        Parameters
        ---------
        
        scale: float, default 1.0
            value with which to scale the datasource shape
            
        Notes
        ---------
        for cases w/ all real... this is pretty simple
        w/ nulls
            there is some bleeding for 'average' methods
            so we need to build a new mask and re-apply w/ numpy
                for this, we use a mask which is null ONLY when all child cells are null
            
                alternatively, we could set null when ANY child cell is null
                
            WARNING: for upsample/aggregate... we don't know how null values are treated
                (i.e., included in the denom or not)
                better to use numpy funcs
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if name is None: 'resamp_r%i'%scale
        _, log, dataset, _, _ = self._func_kwargs(name = name, **kwargs)        
        if prec is None: prec=self.prec        
        log.info('on %s w/ %s'%(dataset.name, dict(resampling=resampling, scale=scale)))
        """
        dataset.read(1)
        dataset.read_masks(1)
        load_array(dataset)
        """
 
        #===========================================================================
        # # resample data to target shape
        #===========================================================================
         
        out_shape=(dataset.count,int(dataset.height * scale),int(dataset.width * scale))
        log.debug('transforming from %s to %s'%(dataset.shape, out_shape))
        
        data_rsmp = dataset.read(1,
            out_shape=out_shape,
            resampling=resampling
                                )
        
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data_rsmp.shape[-1]),
            (dataset.height / data_rsmp.shape[-2])
        )
        
 
        #===========================================================================
        # resample nulls
        #===========================================================================
        """opaque handling of nulls
        msk_rsmp = dataset.read_masks(1, 
                out_shape=out_shape,
                resampling=Resampling.nearest, #doesnt bleed
            )""" 
        mar_raw = dataset.read_masks(1)
        #downsample.disag. (zoom in)
        if scale>1.0:
            #msk_rsmp = skimage.transform.resize(mar_raw, (out_shape[1], out_shape[2]), order=0, mode='constant')
            msk_rsmp = scipy.ndimage.zoom(mar_raw, scale, order=0, mode='reflect',   grid_mode=True)
 
 
        #upsample. aggregate (those with ALL nulls)
        else:
            """see also  hp.np.apply_block_reduce"""
            #stack windows into axis 1 and 3
            downscale = int(1//scale)
            mar1 = mar_raw.reshape(mar_raw.shape[0]//downscale, downscale, mar_raw.shape[1]//downscale, downscale)
            
 
            #those where the max of the children equals exactly the null value
            msk_rsmp = np.where(np.max(mar1, axis=(1,3))==0, 0,255)
 
 
        
        #===============================================================================
        # coerce transformed nulls
        #===============================================================================
        """needed as some resampling methods bleed out
        theres a few ways to handle this... 
            here we manipulate the data values directly.. which seems the cleanest
        2022-09-08: switched to masked arrays
            
            """
            
        
        #=======================================================================
        # assert data_rsmp.shape==msk_rsmp.shape
        # data_rsmp_f1 = np.where(msk_rsmp==0,  dataset.nodata, data_rsmp).astype(dataset.dtypes[0])
        #=======================================================================
        
        #numpy  mask
        res_mar = ma.array(data_rsmp, mask=np.where(msk_rsmp==0, True, False), fill_value=dataset.nodata)
        
        log.info('resampled from %s to %s'%(dataset.shape, data_rsmp.shape))
        
 
        return self.load_memDataset(res_mar, transform=transform, name=name, logger=log, nodata=dataset.nodata,
                                    crs=dataset.crs, masked=True)
    
    def crop(self,window,
 
             **kwargs):
        """crop datasource to window and write
        
        
        Notes
        -----------
        couldnt find any help with this
        
        not sure how this will work if the window has an offset
        """
        
        
        #=======================================================================
        # defaults
        #=======================================================================
        _, log, dataset, out_dir, ofp = self._func_kwargs(name = 'crop', **kwargs)
        
        #=======================================================================
        # prep data
        #=======================================================================
        #get a window of the data and the masks
        crop_ar = dataset.read(1, window=window)
        crop_mask = dataset.read_masks(1, window=window)
        
        #ensure the nans match the nodata value
        cropM_ar = np.where(crop_mask==0,  dataset.nodata, crop_ar)
                
        #=======================================================================
        # write result
        #=======================================================================
        with rio.open(ofp,'w',
                        driver=self.driver,
                        height=crop_ar.shape[0],
                        width=crop_ar.shape[1],
                        count=self.bandCount,
                        dtype=crop_ar.dtype,
                        crs=dataset.crs,
                        transform=dataset.transform,
                        nodata=dataset.nodata,
                    ) as dst:
            
                dst.write(cropM_ar, indexes=1, 
                              masked=False, #not using numpy.ma
                              )
        
        log.info('cropped %s to %s and wrote to file \n    %s'%(dataset.shape, dst.shape, ofp))
        
        return ofp
    
    def merge(self,ds_name_l,
              merge_kwargs=dict(
                  resampling=Resampling.nearest,
                  method='first',
                  ),
              write=True,
              **kwargs):
        """"
        Parameters
        ---------
        
        ds_name_l : list
            datasets to merge (order matters: see 'method')
            
        merge_kwargs : dict
            argumentst to pass to rasterio.merge.merge
            defaults:
                method='first'
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger, log, dataset, out_dir, ofp = self._func_kwargs(name = 'merge', **kwargs)
        
        log.info('merging on %i'%len(ds_name_l))
        
        #=======================================================================
        # retrieve datasets
        #=======================================================================
        ds_l = [self.dataset_d[n] for n in ds_name_l]
        
        #=======================================================================
        # set base
        #=======================================================================
        if self._base() is None:
            self._base_inherit(ds=ds_l[0])
        
        #=======================================================================
        # execute merge
        #=======================================================================
        
        merge_ar, merge_trans = rasterio.merge.merge(ds_l,  **merge_kwargs)
        merge_ar=merge_ar[0] #single band
        
        """
        merge_ar.shape
        plot_rast(merge_ar[0], nodata=self.nodata)
        """
        

 
        
        #=======================================================================
        # write
        #=======================================================================
        if not write:
            res= merge_ar, merge_trans
        else:
            res =self.write_dataset(merge_ar, ofp=ofp, logger=log, transform=merge_trans)
            
        return res
    
    
    #===========================================================================
    # IO---------
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
        
        
        #=======================================================================
        # check if weve loaded already
        #=======================================================================
        assert not fp in self.dataset_d, fp
        #=======================================================================
        # load
        #=======================================================================
        dataset = rasterio.open(fp, mode='r', **kwargs)
 
        
        #=======================================================================
        # clean up the name
        #=======================================================================
        #=======================================================================
        # try:
        #     dataset.clean_name=os.path.basename(dataset.name).replace('.tif', '')
        # except Exception as e:
        #     log.warning('failed to build clean_name w/ %s'%e)
        #     dataset.clean_name = dataset.name
        #=======================================================================
        
        
        #=======================================================================
        # #meta
        #=======================================================================
        if meta:
 
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
        if dataset.name in self.dataset_d:
            raise IOError('dataset already loaded %s'%(dataset.name))
        self.dataset_d[dataset.name] = dataset
        
        return dataset
    
    

 
    def write_array(self,raw_ar,
                       masked=False,
                       crs=None,nodata=None,transform=None,dtype=None,compress=None,
                       write_kwargs=dict(),
                       **kwargs):
        """write an array to raster using rio"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        _, log, _, _, ofp = self._func_kwargs(name = 'write', **kwargs)
        
        crs, _, _, transform, nodata = self._get_refs(crs=crs, nodata=nodata, transform=transform)
        
        if compress is None: compress=self.compress
        if dtype is None: dtype=raw_ar.dtype
        #=======================================================================
        # precheck
        #=======================================================================
        assert len(raw_ar.shape)==2
 
        assert np.issubdtype(dtype, np.number), 'bad dtype: %s'%dtype.name
        #assert 'float' in data.dtype.name
        
        
        #=======================================================================
        # #handle nulls
        #=======================================================================
        """becuase we usually deal with nulls (instead of raster no data values)
        here we convert back to raster nodata vals before writing to disk"""
        if masked:
            assert isinstance(raw_ar, ma.MaskedArray)
            data = raw_ar
            
            assert raw_ar.mask.shape==raw_ar.shape, os.path.basename(ofp)
        else:
            assert not isinstance(raw_ar, ma.MaskedArray)
            
            if np.any(np.isnan(raw_ar)):
                data = np.where(np.isnan(raw_ar), nodata, raw_ar).astype(dtype)
            else:
                data = raw_ar.astype(dtype)
        #=======================================================================
        # write
        #=======================================================================
        with rasterio.open(ofp,'w',
                driver=self.driver,
                height=raw_ar.shape[0],width=raw_ar.shape[1],
                count=self.bandCount,
                dtype=dtype,crs=crs,transform=transform,nodata=nodata,compress=compress,
                ) as dst:
            

            
            dst.write(data, indexes=1, 
                          masked=masked, #build mask from location of nodata values
                          **write_kwargs)
                
            log.info('wrote %s on crs %s (masked=%s) to \n    %s'%(str(dst.shape), crs, masked, ofp))
        
        return ofp
 
    
    def load_memDataset(self,raw_ar,
                       name='memfile',
                       masked=False,
                       crs=None,nodata=None,transform=None,dtype=None,
                       write_kwargs=dict(),
                       **kwargs):
        """load an array as a memory data source"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        _, log, _, _, ofp = self._func_kwargs(name = 'memDS', **kwargs)
        
        crs, _, _, transform, nodata = self._get_refs(crs=crs, nodata=nodata, transform=transform)
        if dtype is None: dtype=raw_ar.dtype
        #=======================================================================
        # #handle nulls
        #=======================================================================
        """becuase we usually deal with nulls (instead of raster no data values)
        here we convert back to raster nodata vals before writing to disk"""
        if not masked:
            assert not isinstance(raw_ar, ma.MaskedArray)
            if np.any(np.isnan(raw_ar)):
                data = np.where(np.isnan(raw_ar), nodata, raw_ar).astype(dtype)
            else:
                data = raw_ar.astype(dtype)
                
        else:
            assert isinstance(raw_ar, ma.MaskedArray)
            assert raw_ar.fill_value==nodata,'fill_value mismatch'
            if np.all(raw_ar.mask):
                log.warning('fully masked!')
                
            data = raw_ar
            
        
        #=======================================================================
        # build memory data
        #=======================================================================
        memfile= rasterio.io.MemoryFile()
        dataset = memfile.open(crs=crs, transform=transform, nodata=nodata, 
                              height=raw_ar.shape[0],width=raw_ar.shape[1],
                              driver=self.driver,
                              count=self.bandCount, dtype=dtype)
          
        dataset.write(data, indexes=1, 
                          masked=masked, #build mask from location of nodata values
                          **write_kwargs)
        
        #=======================================================================
        # add for cleanup
        #=======================================================================
 
        self.dataset_d[name] = dataset
        self.memoryfile_d[name] = memfile
    
        return dataset
    
    def write_memDataset(self, dataset,
                         masked=False, 
                         compress=None,
                         dtype=None,
                         **kwargs):
        """surprised there is no builtin..."""
        
        _, log, _, _, ofp = self._func_kwargs(name = 'wmemDS', **kwargs)
        if compress is None: compress=self.compress
        
        #=======================================================================
        # extract
        #=======================================================================
        #kwargs from dataset
        profile = {k:getattr(dataset, k) for k in ['height', 'width', 'crs', 'transform', 'nodata']}
        
        data = dataset.read(1, masked=masked)
        if dtype is None: dtype = dataset.dtypes[0]
        #=======================================================================
        # write
        #=======================================================================
        with rasterio.open(ofp,'w',
                driver=self.driver, count=self.bandCount,compress=compress, dtype=dtype,
                **profile) as dst:
            

            
            dst.write(data, indexes=1, 
                          masked=masked, #build mask from location of nodata values
                          )
                
            log.info('wrote %s on crs %s to \n    %s'%(str(dst.shape), dataset.crs, ofp))
        
        return ofp
        
        

    #===========================================================================
    # HELPERS----------
    #===========================================================================
 
    
    def get_ndcnt(self, **kwargs):
        logger, dataset, *args = self._func_kwargs(**kwargs)
        
        msk = dataset.read_masks(1)  #read the GDAL RFC 15 mask
        nodata_cnt = (msk==0).sum()
        
        del msk
        return nodata_cnt
        
    #===========================================================================
    # PRIVATES---------
    #===========================================================================
    def _get_dsn(self, input):
        if not isinstance(input, list):
            input = [input]
            
        res_l = list()
        for fp in input:
            res_l.append(self.open_dataset(fp).name)
            
        return res_l
    
    def _func_kwargs(self, logger=None, dataset=None, out_dir=None, ofp=None,name=None, 
                     tmp_dir=None, write=None):
        """typical default for class functions"""
 
        if logger is None:
            logger=self.logger
 
        
        if not name is None:
            log = logger.getChild(name)
        else:
            log = logger
 
        
        if dataset is None:
            dataset = self._base()
        
 
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
    

    def _base(self):
        if self.ref_name in self.dataset_d:
            return self.dataset_d[self.ref_name]
        else:
            return None
        

    def _clear(self):
        #close all open datasets
        for k,v in self.dataset_d.items():
            v.close()
            
        for k,v in self.memoryfile_d.items():
            v.close()
            
    def __enter__(self):
        return self
    
    def __exit__(self,  *args,**kwargs):
        #print('RioWrkr.__exit__')
        self._clear()
 

            
#===============================================================================
# HELPERS----------
#===============================================================================
def write_array(data,ofp,
                crs=rio.crs.CRS.from_epsg(2953),
                transform=rio.transform.from_origin(0,0,1,1), #dummy identify
                nodata=-9999,
                dtype=None,
                driver='GTiff',
                count=1,
                compress=None,
                masked=False,
                ):
    """skinny array to raster file writer
    
    better to just use the sourcecode
    """
    
    #===========================================================================
    # build init
    #===========================================================================
 
    shape = data.shape
    if dtype is None:
        dtype=data.dtype
 
    #===========================================================================
    # execute
    #===========================================================================
    with rio.open(ofp,'w',driver='GTiff',
                  height=shape[0],width=shape[1],count=count,
                dtype=dtype,crs=crs,transform=transform,nodata=nodata,
                compress=compress,
                ) as dst:            
            dst.write(data, indexes=count,masked=masked)
        
    return ofp

def load_array(rlay_obj, 
               indexes=1,
                 window=None,
                 masked=False,
                 ):
    """skinny array from raster object"""
    
    #retrival function
    def get_ar(dataset):
        raw_ar = dataset.read(indexes, window=window, masked=masked)
        
        if masked:
            ar = raw_ar
            assert isinstance(ar, ma.MaskedArray)
            assert not np.all(ar.mask)
            assert ar.mask.shape==raw_ar.shape
        else:
            #switch to np.nan
            mask = dataset.read_masks(indexes, window=window)
            
            ar = np.where(mask==0, np.nan, raw_ar).astype(dataset.dtypes[0])
            
            #check against nodatavalue
            assert not np.any(ar==dataset.nodata), 'mismatch between nodata values and the nodata mask'
        return ar
    
    #flexible application

    return rlay_apply(rlay_obj, get_ar)

def rlay_apply(rlay, func):
    """flexible apply a function to either a filepath or a rio ds"""
    
    if isinstance(rlay, str):
        with rio.open(rlay, mode='r') as ds:
            res = func(ds)
            
    elif isinstance(rlay, rio.io.DatasetReader) or isinstance(rlay, rio.io.DatasetWriter):
        res = func(rlay)
        
    else:
        raise IOError(type(rlay))
    
    return res

#===============================================================================
# def resample(rlay, ofp, scale=1, resampling=Resampling.nearest):
#     """skinny resampling"""
#     
#     
#     
#     def func(dataset):
#         out_shape=(dataset.count,int(dataset.height * scale),int(dataset.width * scale))
#         
#         data_rsmp = dataset.read(1,
#             out_shape=out_shape,
#             resampling=resampling
#                                 )
#         
#         # scale image transform
#         transform = dataset.transform * dataset.transform.scale(
#             (dataset.width / data_rsmp.shape[-1]),
#             (dataset.height / data_rsmp.shape[-2])
#         )
#         
#  
#         #===========================================================================
#         # resample nulls
#         #===========================================================================
#         msk_rsmp = dataset.read_masks(1, 
#                 out_shape=out_shape,
#                 resampling=Resampling.nearest, #doesnt bleed
#             ) 
#         
#         #===============================================================================
#         # coerce transformed nulls
#         #===============================================================================
#         """needed as some resampling methods bleed out
#         theres a few ways to handle this... 
#             here we manipulate the data values directly.. which seems the cleanest"""
#  
#         assert data_rsmp.shape==msk_rsmp.shape
#         data_rsmp_f1 = np.where(msk_rsmp==0,  dataset.nodata, data_rsmp).astype(dataset.dtypes[0])
#         
#         profile = {k:getattr(dataset, k) for k in ['crs', 'nodata']}
#         
#         return write_array(data_rsmp_f1, ofp,dtype=dataset.dtypes[0],transform=transform, **profile)
#     
#     return rlay_apply(rlay, func)
#===============================================================================
    
    
def is_divisible(rlay, divisor):
    """check if the rlays dimensions are evenly divislbe by the divisor"""
    assert isinstance(divisor, int)
    
    shape = rlay_apply(rlay, lambda x:x.shape)
        
    for dim in shape:
        if dim%divisor!=0:
            return False

    return True

def get_window(ds, bbox):
    """get a well rounded window from a bbox"""
    #buffer 1 pixel  
    bbox1 = sgeo.box(*bbox.buffer(ds.res[0], cap_style=3, resolution=1).bounds)
    
    #build a window and round                   
    window = rasterio.windows.from_bounds(*bbox1.bounds, transform=ds.transform).round_lengths().round_offsets()
    
    #check the bounds
    wbnds = sgeo.box(*rasterio.windows.bounds(window, ds.transform))
    
    assert sgeo.box(*ds.bounds).covers(wbnds), 'bounding box exceeds raster extent'
    
    return window, ds.window_transform(window)
    

def get_stats(ds, att_l=['crs', 'height', 'width', 'transform', 'nodata', 'bounds']):
    d = dict()
    for attn in att_l:
        d[attn] = getattr(ds, attn)
    return d

def get_ds_attr(rlay, stat):
    return rlay_apply(rlay, lambda ds:getattr(ds, stat))

def plot_rast(ar_raw,
              ax=None,
              cmap='gray',
              interpolation='nearest',
              txt_d = None,
 
              transform=None,
              **kwargs):
    """plot a raster array
    see also hp.plot
    TODO: add a histogram"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    if ax is None:
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        limits = None
    else:
        limits = ax.axis()
        
    if txt_d is None: txt_d=dict()
    
    imkwargs = {**dict(cmap=cmap,interpolation=interpolation), **kwargs}
    
    #===========================================================================
    # plot the image
    #===========================================================================
    ax_img = show(ar_raw, transform=transform, ax=ax,contour=False, **imkwargs)
    #ax_img = ax.imshow(masked_ar,cmap=cmap,interpolation=interpolation, **kwargs)
 
    #plt.colorbar(ax_img, ax=ax) #steal some space and add a color bar
    #===========================================================================
    # add some details
    #===========================================================================
    txt_d.update({'shape':str(ar_raw.shape), 'size':ar_raw.size})
 
    ax.text(0.1, 0.9, get_dict_str(txt_d), transform=ax.transAxes, va='top', fontsize=8, color='red')
    
    #===========================================================================
    # wrap
    #===========================================================================
    if not limits is None:
        ax.axis(limits)
    """
    plt.show()
    """
    
    return ax
#===============================================================================
# ASSERTIONS------
#===============================================================================
def assert_rlay_simple(rlay, msg='',): 
    """square pixels with integer size"""
    if not __debug__: # true if Python was not started with an -O option
        return
    
    __tracebackhide__ = True  
    
    #retriever
    d = dict()
    def set_stats(ds):
        for attn in ['shape', 'res']:
            d[attn] = getattr(ds, attn)
        
    #===========================================================================
    # retrieve stats
    #===========================================================================
    rlay_apply(rlay, set_stats)
   
    #===========================================================================
    # check
    #===========================================================================
 
    x, y = d['res']
 
    
    if not x==y:
        raise AssertionError('non-square pixels\n' + msg)
 
    if not round(x, 10)==int(x):
        raise AssertionError('non-integer pixel size\n' + msg)
    
def assert_extent_equal(left, right, msg='',): 
    """ extents check"""
    if not __debug__: # true if Python was not started with an -O option
        return
 
    __tracebackhide__ = True
    
    def get_stats(ds):
        return {'bounds':ds.bounds, 'crs':ds.crs}
    
    ld = rlay_apply(left, get_stats)
    rd = rlay_apply(right, get_stats)
    #===========================================================================
    # crs
    #===========================================================================
    if not ld['crs']==rd['crs']:
        raise AssertionError('crs mismatch')
    #===========================================================================
    # extents
    #===========================================================================
    le, re = ld['bounds'], rd['bounds']
    if not le==re:
        raise AssertionError('extent mismatch \n    %s != %s\n    '%(
                le, re) +msg) 

def assert_ds_attribute_match(rlay,
                          crs=None, height=None, width=None, transform=None, nodata=None,bounds=None,
                          msg=''):

    
    stats_d = rlay_apply(rlay, get_stats)
    
    chk_d = {'crs':crs, 'height':height, 'width':width, 'transform':transform, 'nodata':nodata, 'bounds':bounds}
    
    cnt=0
    for k, cval in chk_d.items():
        if not cval is None:
            if not cval==stats_d[k]:
                raise AssertionError('stat \'%s\' does not meet passed expectation (%s vs. %s) \n '%(
                    k, cval, stats_d[k])+msg)
            cnt+=1
    
    if not cnt>0:
        raise IOError('no check values passed')
 
    
    
             
                          
        
        
        
        
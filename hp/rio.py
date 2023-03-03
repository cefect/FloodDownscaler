'''
Created on Aug. 7, 2022

@author: cefect
'''
import os, warnings
import numpy as np
 
import numpy.ma as ma
import rasterio as rio
import shapely.geometry as sgeo
from shapely.geometry.polygon import Polygon
 
#print('rasterio.__version__:%s'%rio.__version__)

assert os.getenv('PROJ_LIB') is None, 'rasterio expects no PROJ_LIB but got \n%s'%os.getenv('PROJ_LIB')
 
import rasterio.merge
import rasterio.io
import rasterio.features
from rasterio.plot import show
from rasterio.enums import Resampling, Compression

import fiona #not a rasterio dependency? needed for aoi work
from pyproj.crs import CRS

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
                compress=Compression('NONE'),nodata=-9999, 
                 
                 #reference inheritance
                 #crs=None,height=None,width=None,transform=None,nodata=None,
                 
 
                 
                 **kwargs):

 
        super().__init__(**kwargs)
        
 
        
        #=======================================================================
        # simple attachments
        #=======================================================================
        self.compress=compress
        
        assert isinstance(compress, Compression)
        self.nodata=nodata
 
 
        self.dataset_d = dict() #all loaded datasets
        self.memoryfile_d=dict()
        #=======================================================================
        # set reference
        #=======================================================================        
        if not rlay_ref_fp is None:
            self._base_set(rlay_ref_fp)
            
        #=======================================================================
        # inherit properties from reference 
        #=======================================================================
        #pars_d=self._base_inherit(crs=crs, height=height, width=width, transform=transform, nodata=nodata)
        
        #self.logger.debug('init w/ %s'%pars_d)
        
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
                    #self.logger.debug('no value passed for %s'%attName)
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
    
    def _set_defaults(self, d):
        for attn in ['crs', 'height', 'width', 'transform', 'nodata']:
            assert attn in d, attn
            setattr(self, attn, d[attn])
            
                   
    

    
    #===========================================================================
    # MANIPULATORS----------
    #===========================================================================
    
    def resample(self,
                 dataset,
                 resampling=Resampling.nearest,
                 scale=1.0,
                 #write=True,
                 #update_ref=False, 
 
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
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('resample',  **kwargs)
     
         
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
        
        """
        crop_ar.shape
        """
                
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
                        transform=dataset.window_transform(window),
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
                       crs=None,nodata=None,transform=None,dtype=None,compress=None,driver=None,bandCount=None,
                       width=None, height=None, 
                       **kwargs):
        """write an array to raster using rio using session defaults"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('write_array',ext='.tif', **kwargs)
        
        crs, _, _, transform, nodata = self._get_refs(crs=crs, nodata=nodata, transform=transform)
        
        if compress is None: compress=self.compress
        if dtype is None: dtype=raw_ar.dtype
        if driver is None: driver=self.driver
        if bandCount is None: bandCount=self.bandCount

        
        kwargs2 = dict(masked=masked, crs=crs, transform=transform,nodata=nodata, 
                       dtype=dtype, compress=compress, driver=driver, count=bandCount,
                       width=width, height=height)
 
        _ = write_array(raw_ar, ofp, **kwargs2)
                         
        

        log.info(f'wrote {str(raw_ar.shape)} on crs {crs} (masked={masked}) to \n    {ofp}')
        
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
    # VECTORS------
    #===========================================================================
    """ see hp.gpd"""

        

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

        
        #return rlay_kwargs
        
    def _get_dsn(self, input):
        if not isinstance(input, list):
            input = [input]
            
        res_l = list()
        for fp in input:
            res_l.append(self.open_dataset(fp).name)
            
        return res_l
    
 #==============================================================================
 #    def _func_kwargs(self, logger=None, dataset=None, out_dir=None, ofp=None,name=None, 
 #                     tmp_dir=None, write=None):
 #        """typical default for class functions"""
 # 
 #        if logger is None:
 #            logger=self.logger
 # 
 #        
 #        if not name is None:
 #            log = logger.getChild(name)
 #        else:
 #            log = logger
 # 
 #        
 #        if dataset is None:
 #            dataset = self._base()
 #        
 # 
 #        if out_dir is None:
 #            out_dir=self.out_dir
 #            
 #        if ofp is None:
 #            if name is None:
 #                ofp = os.path.join(out_dir, self.fancy_name + '.tif')
 #            else:
 #                ofp = os.path.join(out_dir, self.fancy_name + '_%s.tif'%name)
 #            
 #            
 #        return logger, log, dataset, out_dir, ofp
 #==============================================================================
    
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
        

 
        
class RioSession(RioWrkr):
    aoi_fp=None
    
    def __init__(self, 
                 #==============================================================
                 # crs=CRS.from_user_input(25832),
                 # bbox=
                 #==============================================================
                 crs=None, bbox=None, aoi_fp=None,
                 
                 #defaults
                 
                 
                 **kwargs):
        
        """"
        
        Parameters
        -----------
        
        bbox: shapely.polygon
            bounds assumed to be on the same crs as the data
            sgeo.box(0, 0, 100, 100),
            
        crs: <class 'pyproj.crs.crs.CRS'>
            coordinate reference system
        """
        super().__init__(**kwargs)
        
        #=======================================================================
        # set aoi
        #=======================================================================
        if not aoi_fp is None:            
            assert crs is None
            assert bbox is None
            self._set_aoi(aoi_fp)
            

    def __enter__(self):
        return self
    
    def __exit__(self,  *args,**kwargs):
        #print('RioWrkr.__exit__')
        self._clear()
        
class RioSession(RioWrkr):
    aoi_fp=None
    
    def __init__(self, 
                 #==============================================================
                 # crs=CRS.from_user_input(25832),
                 # bbox=
                 #==============================================================
                 crs=None, bbox=None, aoi_fp=None,
                 
                 #defaults
                 
                 
                 **kwargs):
        
        """"
        
        Parameters
        -----------
        
        bbox: shapely.polygon
            bounds assumed to be on the same crs as the data
            sgeo.box(0, 0, 100, 100),
            
        crs: <class 'pyproj.crs.crs.CRS'>
            coordinate reference system
        """
        super().__init__(**kwargs)
        
        #=======================================================================
        # set aoi
        #=======================================================================
        if not aoi_fp is None:            
            assert crs is None
            assert bbox is None
            self._set_aoi(aoi_fp)
            

        else:
            self.crs=crs
            self.bbox = bbox
            
        #check
        if not self.crs is None:
            assert isinstance(self.crs, CRS)
            
        if not self.bbox is None:
            assert isinstance(self.bbox, Polygon)
 
        
        
    def _set_aoi(self, aoi_fp):
        assert os.path.exists(aoi_fp)
        
        #open file and get bounds and crs using fiona
        with fiona.open(aoi_fp, "r") as source:
            bbox = sgeo.box(*source.bounds) 
            crs = CRS(source.crs['init'])
            
        self.crs=crs
        self.bbox = bbox
        
        self.logger.info('set crs: %s'%crs.to_epsg())
        self.aoi_fp=aoi_fp
        
        return self.crs, self.bbox
    
    def _get_defaults(self, crs=None, bbox=None, nodata=None, compress=None,
                      as_dict=False):
        """return session defaults for this worker
        
        EXAMPLE
        ----------
        crs, bbox, compress, nodata =RioSession._get_defaults(self)
        """
        if crs is None: crs=self.crs
        if bbox is  None: bbox=self.bbox
        if compress is None: compress=self.compress
        if nodata is None: nodata=self.nodata
        
        if not as_dict:
            return crs, bbox, compress, nodata
        else:
            return dict(crs=crs, bbox=bbox, compress=compress, nodata=nodata)

            
#===============================================================================
# HELPERS----------
#===============================================================================

 
def write_array2(ar, ofp, 
                 count=1, width=None, height=None, nodata=-9999, dtype=None,
                 **kwargs):
 
    """skinny writer"""
    #===========================================================================
    # precheck
    #===========================================================================
    assert isinstance(ofp, str), ofp
    assert os.path.exists(os.path.dirname(ofp)), ofp
    
    #===========================================================================
    # defaults
    #===========================================================================
    if width is None or height is None:
        height, width  = ar.shape
        
    if dtype is None:
        dtype=ar.dtype
    
    #===========================================================================
    # write
    #===========================================================================
    with rio.open(ofp, 'w', 
                  count=count, width=width, height=height,nodata=nodata, dtype=dtype,
                  **kwargs) as ds:
        ds.write(ar, indexes=1, masked=False)
    return ofp
            

def write_array(raw_ar,ofp,
                crs=rio.crs.CRS.from_epsg(2953),
                transform=rio.transform.from_origin(0,0,1,1), #dummy identify
                nodata=-9999,
                dtype=None,
                driver='GTiff',
                count=1,
                compress=None,
                masked=False,
                width=None,
                height=None,
                **kwargs):
    """array to raster file writer with nodata handling and transparent defaults
    
    Parameters
    ----------

    raw_ar: np.Array
        takes masked or non-masked. the latter is converted to a masked before writing
        

    masked: bool default False
        if True, the result usually has 2 bands
    """
    
    #===========================================================================
    # build init
    #===========================================================================
 
    shape = raw_ar.shape
    if dtype is None:
        dtype=raw_ar.dtype
        
    if width is None:
        width=shape[1]
    if height is None:
        height=shape[0]
    
    #===========================================================================
    # precheck
    #===========================================================================
    if os.path.exists(ofp):
        os.remove(ofp)
        
    assert len(raw_ar.shape)==2
    
    assert np.issubdtype(dtype, np.number), 'bad dtype: %s'%dtype.name
    
    #=======================================================================
    # #handle nulls
    #=======================================================================
    """becuase we usually deal with nulls (instead of raster no data values)
    here we convert back to raster nodata vals before writing to disk"""
 
    if isinstance(raw_ar, ma.MaskedArray):
        data = raw_ar        
        assert raw_ar.mask.shape==raw_ar.shape, os.path.basename(ofp)
        
    elif isinstance(raw_ar, np.ndarray):
        if np.any(np.isnan(raw_ar)):
            data = np.where(np.isnan(raw_ar), nodata, raw_ar).astype(dtype)
        else:
            data = raw_ar.astype(dtype)
    
    else:
        raise TypeError(type(raw_ar))
 
    #===========================================================================
    # execute
    #===========================================================================

    print(f'writing {data.shape} to {ofp}')

    with rio.open(ofp,'w',driver=driver,
                  height=height,width=width,
                  count=count,dtype=dtype,crs=crs,transform=transform,nodata=nodata,compress=compress,
                 **kwargs) as dst:            
            dst.write(data, indexes=count,
                      masked=masked,
                      #we do this explicitly above

                      #If given a Numpy MaskedArray and masked is True, the inputï¿½s data and mask will be written to the datasetï¿½s bands and band mask. 
                     #If masked is False, no band mask is written. Instead, the input arrayï¿½s masked values are filled with the datasetï¿½s nodata value (if defined) or the inputï¿½s own fill value.

                      )
            
        
    return ofp



def load_array(rlay_obj, 
               indexes=1,
                 window=None,
                 masked=True,
                 bbox=None,
                 ):
    """skinny array from raster object"""
    
    if window is not None:
        assert bbox is None
 
    
    #retrival function
    def get_ar(dataset):
        if bbox is not None:
            window1 =  rio.windows.from_bounds(*bbox.bounds, transform=dataset.transform)
        else:
            window1 = window

        
        
        raw_ar = dataset.read(indexes, window=window1, masked=masked)
        
        if masked:
            ar = raw_ar
            assert isinstance(ar, ma.MaskedArray)
            assert not np.all(ar.mask)
            assert ar.mask.shape==raw_ar.shape
        else:
            
            #switch to np.nan
            mask = dataset.read_masks(indexes, window=window1)
            
            bx = mask==0
            if bx.any():
                assert 'float' in dataset.dtypes[0], 'unmaked arrays not supported for non-float dtypes'
            
                ar = np.where(bx, np.nan, raw_ar).astype(dataset.dtypes[0])
            else:
                ar=raw_ar.astype(dataset.dtypes[0])
            
            #check against nodatavalue
            assert not np.any(ar==dataset.nodata), 'mismatch between nodata values and the nodata mask'
        return ar
    
    #flexible application

    return rlay_apply(rlay_obj, get_ar)


def rlay_apply(rlay, func, **kwargs):
    """flexible apply a function to either a filepath or a rio ds"""
    
    assert not rlay is None
    
    if isinstance(rlay, str):
        with rio.open(rlay, mode='r') as ds:
            res = func(ds, **kwargs)
            
    elif isinstance(rlay, rio.io.DatasetReader) or isinstance(rlay, rio.io.DatasetWriter):
        res = func(rlay, **kwargs)
        
    else:
        raise IOError(type(rlay))
    
    return res

 
def rlay_ar_apply(rlay, func, masked=True, **kwargs):
 
    """apply a func to an array
    
    takes a function like
        f(np.Array, **kwargs)
    """

    def ds_func(dataset, **kwargs):
        return func(dataset.read(1, window=None, masked=masked), **kwargs)
    
    return rlay_apply(rlay, ds_func, **kwargs)
        

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
    
    


def get_window(ds, bbox,
                round_offsets=False,
                 round_lengths=False,
                 ):
    """get a well rounded window from a bbox"""
    #buffer 1 pixel  
    bbox1 = sgeo.box(*bbox.buffer(ds.res[0], cap_style=3, resolution=1).bounds)
    
    #build a window and round                   
    window = rasterio.windows.from_bounds(*bbox1.bounds, transform=ds.transform)
    
    if round_offsets:
        window = window.round_offsets()
        
    if round_lengths:
        window = window.round_lengths()
    
 
    
    #check the bounds
    wbnds = sgeo.box(*rasterio.windows.bounds(window, ds.transform))
    
    assert sgeo.box(*ds.bounds).covers(wbnds), 'bounding box exceeds raster extent'
    
    return window, ds.window_transform(window)
    

def get_stats(ds, att_l=['crs', 'height', 'width', 'transform', 'nodata', 'bounds', 'res', 'dtypes']):
    d = dict()
    for attn in att_l:        
        d[attn] = getattr(ds, attn)
    return d

def get_stats2(rlay, **kwargs):

    warnings.warn("deprecated (2023 01 28). use get_meta() instead", DeprecationWarning)
    return rlay_apply(rlay, lambda x:get_stats(x, **kwargs))

def get_meta(rlay, **kwargs):

    return rlay_apply(rlay, lambda x:get_stats(x, **kwargs))

def get_ds_attr(rlay, stat):
    return rlay_apply(rlay, lambda ds:getattr(ds, stat))

def get_profile(rlay):
    return rlay_apply(rlay, lambda ds:ds.profile)

def get_write_kwargs( obj,
                      att_l = ['crs', 'transform', 'nodata'],
                      **kwargs):
    """convenience for getting write kwargsfrom datasource stats"""
    #=======================================================================
    # load from filepath
    #=======================================================================
    if isinstance(obj, str) or isinstance(obj,rasterio.io.DatasetReader):
        stats_d  = rlay_apply(obj, get_stats, att_l=att_l+['dtypes'])
    elif isinstance(obj, dict):
        stats_d = obj

    else:
        raise TypeError(type(obj))
                        
    rlay_kwargs = {**kwargs,
        **{k:stats_d[k] for k in att_l}}  
          
    #handle tuple
    rlay_kwargs['dtype'] = stats_d['dtypes'][0]
    
    return rlay_kwargs


def rlay_calc1(rlay_fp, ofp, statement):
    """evaluate a statement with numpy math on a single raster"""
    
    with rio.open(rlay_fp, mode='r') as ds:
        ar = load_array(ds)
        
        result = statement(ar)
        
        assert isinstance(result, np.ndarray)
        
        profile = ds.profile
        
    #write
    with rio.open(ofp, mode='w', **profile) as dest:
        dest.write(
            np.where(np.isnan(result), profile['nodata'],result),
             1)
    
    return ofp
        
    
        
    

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

def get_xy_coords(transform, shape):
    """return an array of spatial values for x and y
    
    surprised there is no builtin
    
    this is needed  by xarray
    
    print(f'x, cols:{s[1]}    y, rows:{s[0]}')
    """
    transformer = rio.transform.AffineTransformer(transform) 
    x_ar, _ = transformer.xy(np.full(shape[1], 0), np.arange(shape[1])) #rows, cols            
    _, y_ar = transformer.xy(np.arange(shape[0]), np.full(shape[0], 0)) #rows, cols
    
    return x_ar, y_ar

def get_depth(dem_fp, wse_fp, ofp=None):
    """add dem and wse to get a depth grid"""
    
    assert_spatial_equal(dem_fp, wse_fp)
    
    if ofp is None:
        fname = os.path.splitext( os.path.basename(wse_fp))[0] + '_wsh.tif'
        ofp = os.path.join(os.path.dirname(wse_fp),fname)
    
    #===========================================================================
    # load
    #===========================================================================
    dem_ar = load_array(dem_fp, masked=True)
    
    wse_ar = load_array(wse_fp, masked=True)
    
    #logic checks
    assert not dem_ar.mask.any()
    assert wse_ar.mask.any()
    assert not wse_ar.mask.all()
    
    #===========================================================================
    # calc
    #===========================================================================
    #simple subtraction
    wd1_ar = wse_ar - dem_ar
    
    #identify dry
    dry_bx = np.logical_or(
        wse_ar.mask, wse_ar.data<dem_ar.data
        )
    
    assert not dry_bx.all().all()
    
    #rebuild
    wd2_ar = np.where(~dry_bx, wd1_ar.data, 0.0)
    
    
    #check we have no positive depths on the wse mask
    assert not np.logical_and(wse_ar.mask, wd2_ar>0.0).any()
    
    #===========================================================================
    # write
    #===========================================================================
    
    #convert to masked
    wd2M_ar = ma.array(wd2_ar, mask=np.isnan(wd2_ar), fill_value=wse_ar.fill_value)
    
    assert not wd2M_ar.mask.any(), 'depth grids should have no mask'
    
    return write_array(wd2M_ar, ofp, masked=False, **get_profile(wse_fp))
    
#===============================================================================
# Building New Rasters--------
#===============================================================================
def write_resample(rlay_fp,
                 resampling=Resampling.nearest,
                 scale=1.0,
                 #write=True,
                 #update_ref=False, 
 
                 ofp=None,out_dir=None,
                 #**kwargs,
                 ):
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

                               
        #===========================================================================
        # # resample data to target shape
        #===========================================================================
        with rasterio.open(rlay_fp, mode='r') as dataset:
            
            """
            dataset.read(1, masked=False)
            """
         
            out_shape=(dataset.count,int(dataset.height * scale),int(dataset.width * scale))
 
            
            data_rsmp = dataset.read(1,
                out_shape=out_shape,
                resampling=resampling,
                masked=True,
                                    )
        
            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data_rsmp.shape[-1]),
                (dataset.height / data_rsmp.shape[-2])
            )
        
 
            outres = dataset.res[0]/scale
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
                mar_raw.shape
                #stack windows into axis 1 and 3
 
                downscale = int(1/scale)
                mar1 = mar_raw.reshape(mar_raw.shape[0]//downscale, 
                                       downscale, 
                                       mar_raw.shape[1]//downscale, 
                                       downscale)
                
     
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
            
            #===================================================================
            # write
            #===================================================================
            if out_dir is None:
                out_dir = os.path.dirname(rlay_fp)
            assert os.path.exists(out_dir)
            if ofp is None:
                fname, ext = os.path.splitext(os.path.basename(rlay_fp))                
                ofp = os.path.join(out_dir,f'{fname}_r{outres}{ext}')
            
            #build new profile
            prof_rsmp = {**dataset.profile, 
                      **dict(
                          width=data_rsmp.shape[-1], 
                          height=data_rsmp.shape[-2],
                          transform=transform,
                          )}
            

            return write_array2(res_mar,ofp, **prof_rsmp)

            


def write_clip(raw_fp, 
                window=None,
                bbox=None,
                 
                masked=True,
                 crs=None, 
 
                 ofp=None,
                 **kwargs):
    """write a new raster from a window"""

    
    with rio.open(raw_fp, mode='r') as ds:
        
        #crs check/load
        if not crs is None:
            assert crs==ds.crs
        else:
            crs = ds.crs
        
        #window default
        if window is None:
            window = rasterio.windows.from_bounds(*bbox.bounds, transform=ds.transform)
 
        else: 
            assert bbox is None
            
        #get the windowed transform
        transform = rasterio.windows.transform(window, ds.transform)
        
        #get stats
        stats_d = get_stats(ds)
        stats_d['bounds'] = rio.windows.bounds(window, transform=transform)
            
        #load the windowed data
        ar = ds.read(1, window=window, masked=masked)
        
        #=======================================================================
        # #write clipped data
        #=======================================================================
        if ofp is None:
            fname = os.path.splitext( os.path.basename(raw_fp))[0] + '_clip.tif'
            ofp = os.path.join(os.path.dirname(raw_fp),fname)
        
        write_kwargs = get_write_kwargs(ds)
        write_kwargs1 = {**write_kwargs, **dict(transform=transform), **kwargs}
        
        ofp = write_array(ar, ofp,  masked=False,   **write_kwargs1)
        
    return ofp, stats_d



def write_mask_apply(rlay_fp, mask_ar,
 
                     logic=np.logical_or,
               ofp=None,out_dir=None,
               ):
    """mask the passed rlay by the passed mask
    
    NOTE: using numpy mask convention (True=Mask)
    
    Parameters
    -----------
    mask_ar: np.array
        boolean mask
        
    logic: function or None
        numpy logic function to apply to the raw mask and the new mask
            e.g., set the mask as mask values in either raster
        
        or
        
        None: just use the new mask
            
    """
    
    #assert_spatial_equal(rlay_fp, mask_fp)
 
    assert isinstance(mask_ar, np.ndarray)
    assert mask_ar.dtype==np.dtype('bool')
    assert not np.all(mask_ar)
    assert np.any(mask_ar)
    #===========================================================================
    # retrieve raw
    #===========================================================================
    with rio.open(rlay_fp, mode='r') as dataset:
        
        raw_ar = dataset.read(1, window=None, masked=True)
        
        profile = dataset.profile
        
 
    
    #===========================================================================
    # apply mask
    #===========================================================================
    if not logic is None:
        new_mask_ar = logic(raw_ar.mask, mask_ar)
    else:
        new_mask_ar = mask_ar
        
    assert mask_ar.dtype==np.dtype('bool')
    if not np.any(mask_ar):
        raise Warning('no masked values!')
    
    #===========================================================================
    # rebuild 
    #===========================================================================
    new_ar = ma.array(raw_ar.data, mask=new_mask_ar)
    
    #===========================================================================
    # write
    #===========================================================================
    if out_dir is None:
        out_dir = os.path.dirname(rlay_fp)
    assert os.path.exists(out_dir)
    
    if ofp is None:
        fname, ext = os.path.splitext(os.path.basename(rlay_fp))                
        ofp = os.path.join(out_dir,f'{fname}_maskd{ext}')
                
    
    
    return write_array(new_ar, ofp,  masked=False,   **profile)
    
    
def write_mosaic(fp1, fp2, ofp=None):
    """combine valid cell values on two rasters"""
    
    #===========================================================================
    # load
    #===========================================================================
    assert_spatial_equal(fp1, fp2)
    ar1 = load_array(fp1, masked=True)
    
    ar2 = load_array(fp2, masked=True)
    
    #===========================================================================
    # check overlap
    #===========================================================================
    overlap = np.logical_and(~ar1.mask, ~ar2.mask)
    assert not np.any(overlap), f'masks overlap {overlap.sum()}'
    
    
    merge_ar = ma.array(ar1.filled(1)*ar2.filled(1), mask=ar1.mask*ar2.mask)
    
    #===========================================================================
    # write
    #===========================================================================
    return write_array2(merge_ar, ofp, **get_profile(fp1))
    

    
    with rio.open(raw_fp, mode='r') as ds:
        
        #crs check/load
        if not crs is None:
            assert crs==ds.crs
        else:
            crs = ds.crs
        
        #window default
        if window is None:
            window = rasterio.windows.from_bounds(*bbox.bounds, transform=ds.transform)
 
        else: 
            assert bbox is None
            
        #get the windowed transform
        transform = rasterio.windows.transform(window, ds.transform)
        
        #get stats
        stats_d = get_stats(ds)
        stats_d['bounds'] = rio.windows.bounds(window, transform=transform)
            
        #load the windowed data
        ar = ds.read(1, window=window, masked=masked)
        
        #=======================================================================
        # #write clipped data
        #=======================================================================
        if ofp is None:
            fname = os.path.splitext( os.path.basename(raw_fp))[0] + '_clip.tif'
            ofp = os.path.join(os.path.dirname(raw_fp),fname)
        
        write_kwargs = get_write_kwargs(ds)
        write_kwargs1 = {**write_kwargs, **dict(transform=transform), **kwargs}
        
        ofp = write_array(ar, ofp,  masked=False,   **write_kwargs1)
        
    return ofp, stats_d


def rlay_to_polygons(rlay_fp, convert_to_binary=True,
                          ):
    """
    get shapely polygons for each clump in a raster
    
    Parameters
    -----------
    convert_to_binary: bool, True
        polygonize the mask (rather than groups of data values)
        
    """
    
    #===========================================================================
    # collect polygons
    #===========================================================================
    with rio.open(rlay_fp, mode='r') as src:
        mar = src.read(1, masked=True)
        
        if convert_to_binary:
            source = np.where(mar.mask, int(mar.fill_value),1)
        else:
            source = mar
        #mask = image != src.nodata
        d=dict()
        for geom, val in rasterio.features.shapes(source, mask=~mar.mask, transform=src.transform,
                                                  connectivity=8):
            
            d[val] = sgeo.shape(geom)
            
        #print(f'finished w/ {len(d)} polygon')
        
 
        
    return d

 

#===============================================================================
# TESTS--------
#===============================================================================
def is_divisible(rlay, divisor):
    """check if the rlays dimensions are evenly divislbe by the divisor"""
    assert isinstance(divisor, int)
    
    shape = rlay_apply(rlay, lambda x:x.shape)
        
    for dim in shape:
        if dim%divisor!=0:
            return False

    return True

def is_raster_file(filepath):
    """probably some more sophisticated way to do this... but I always use tifs"""
    _, ext = os.path.splitext(filepath)
    return ext in ['.tif']


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
    
def assert_extent_equal(left, right,  msg='',): 
    """ extents check"""
    if not __debug__: # true if Python was not started with an -O option
        return
 
    __tracebackhide__ = True
    
    f= lambda ds, att_l=['crs',  'bounds']:get_stats(ds, att_l=att_l) 
    
    ld = rlay_apply(left, f)
    rd = rlay_apply(right, f)
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

def assert_spatial_equal(left, right,  msg='',): 
    """check all spatial attributes match"""
    if not __debug__: # true if Python was not started with an -O option

        return 

    __tracebackhide__ = True
    
    f= lambda ds, att_l=['crs', 'height', 'width', 'bounds', 'res']:get_stats(ds, att_l=att_l) 
    
    ld = rlay_apply(left, f)
    rd = rlay_apply(right, f)
    
    #===========================================================================
    # check
    #===========================================================================
    for k, lval in ld.items():
        rval = rd[k]
        
        if not lval==rval:
            raise AssertionError(f'{k} mismatch\n    right={rval}\n    left={lval}\n'+msg)
 
        
        
        
     

def assert_ds_attribute_match(rlay,
                          crs=None, height=None, width=None, transform=None, nodata=None,bounds=None,
                          msg=''):

    #assertion setup
    if not __debug__: # true if Python was not started with an -O option
        return
    __tracebackhide__ = True
    
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
 

 
def assert_masked_ar(ar, msg=''):
    """check the array satisfies expectations for a masked array
    
    NOTE: to call this on a raster filepath, wrap with rlay_ar_apply:
        rlay_ar_apply(wse1_dp_fp, assert_wse_ar, msg='result WSe')
    """
    if not __debug__: # true if Python was not started with an -O option
        return
    
    if not isinstance(ar, ma.MaskedArray):
        raise AssertionError(msg+'\n     bad type ' + str(type(ar)))
    if not 'float' in ar.dtype.name:
        raise AssertionError(msg+'\n     bad dtype ' + ar.dtype.name)
    
    #check there are no nulls on the data
    if np.any(np.isnan(ar.filled())):
        raise AssertionError(msg+f'\n    got {np.isnan(ar.data).sum()}/{ar.size} nulls outside of mask')
        
 

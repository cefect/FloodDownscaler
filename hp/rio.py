'''
Created on Aug. 7, 2022

@author: cefect
'''
import os, warnings, tempfile, shutil
import numpy as np
 
import numpy.ma as ma
import rasterio as rio
import shapely.geometry as sgeo

 
#print('rasterio.__version__:%s'%rio.__version__)

assert os.getenv('PROJ_LIB') is None, 'rasterio expects no PROJ_LIB but got \n%s'%os.getenv('PROJ_LIB')
 
import rasterio.merge
import rasterio.io
import rasterio.features
from affine import Affine
from rasterio.plot import show
from rasterio.enums import Resampling, Compression

import fiona #not a rasterio dependency? needed for aoi work
#from pyproj.crs import CRS
from rasterio.crs import CRS

import scipy.ndimage
#import skimage
#from skimage.transform import downscale_local_mean

#import hp.gdal
from hp.oop import Basic
from hp.basic import dstr
from hp.fiona import SpatialBBOXWrkr
#from hp.plot import plot_rast #for debugging
#import matplotlib.pyplot as plt

#===============================================================================
# vars
#===============================================================================
confusion_codes = {'TP':111, 'TN':110, 'FP':101, 'FN':100}

#===============================================================================
# classes
#===============================================================================


class RioWrkr(object):
    """work session for single band raster calcs
    
    RioWrkr vs. RioSession
        RioWrkr: single raster calculations (roughly equivalent to rasterio.datasource)
        
        RioSession: session level calculations
    
    WARNING:
        been difficult to have a consistent structure here.
        decided to make skinny and just match rio.datasource
        
        removed reference to oop.base to avoid clashing during complex inheritance
    
    
    """
    
    #extras for writing
    compress=None
    
 
    #spatial profile
    """nice to have these on the class for more consistent retrival... evven if empty"""
    driver=None
    dtype=None
    nodata=None
    width=None
    height=None
    count=None
    crs=None
    transform=None
    blockysize=None
    tiled=None
    
    profile_expect_d = dict( #type expectations
        driver=str,
        dtype=str,
        nodata=None,
        width=int,
        height=int,
        count=int,
        crs=CRS,
        transform=Affine,
        blockysize=int,
        tiled=bool,
        )
        
 


    def __init__(self,                
                #spatial profile
                profile=dict(
                    nodata=-9999,
                    driver='GTiff',
                    ), 
                rlay_ref_fp=None,
 
                 **kwargs):
        """init a rio datasource worker
        
        

        
        Pars
        -------------
        profile: dict
            spatial meta (overwrites values extracted from reference layer)
            
        rlay_ref_fp: str
            filepath to a reference raster from which to extract spatial meta
            
        """

 
        super().__init__(**kwargs)
 
        
        #=======================================================================
        # profile
        #=======================================================================
        self._set_profile(rlay_ref_fp=rlay_ref_fp, **profile)
 
 
 
    def _set_profile(self, rlay_ref_fp=None, **kwargs):
        """set the spatial profile of this worker from a raster file"""
        
        #pull from reference
        if not rlay_ref_fp is None:
            d1 = get_profile(rlay_ref_fp)
        else:
            d1 = dict()
            
        #update w/ kwargs
        d1.update(kwargs)
        
        #attach
        for k,v in d1.items():
            setattr(self, k, v)
            
            
        """dont want to do this every time... sometimes we init without setting all the atts    
        #check
        self.assert_atts()"""
        
        self.profile={k:getattr(self, k) for k in self.profile_expect_d.keys()}
        
    def get_profile(self):
        """get the datasource profile"""
        self.assert_atts()
        
        #pull fresh from attributes
        self.profile={k:getattr(self, k) for k in self.profile_expect_d.keys()}
        
        return self.profile.copy()
        
        
    def assert_atts(self):
        """check spatial meta"""
        
        for k,v in self.profile_expect_d.items():
            if v is None: continue
            attv = getattr(self, k)
            assert isinstance(attv, v), f'bad type on {k}={attv}\nexpected {v} got {type(attv)}'
            
 
            
        
    def __enter__(self):
        return self
    
    def __exit__(self,  *args,**kwargs):
        pass
        """not needed by lowest object
        super().__exit__(*args, **kwargs)"""
        
 

        
class RioSession(RioWrkr, SpatialBBOXWrkr):
    
    def __init__(self,   
    
                    #default behaviors
                compress=Compression('NONE'), #matching gdal GeoTiff driver naming
                
                **kwargs):
        
        super().__init__(**kwargs)
        
 
        
        #=======================================================================
        # simple attachments
        #=======================================================================
        self.compress=compress        
        assert isinstance(compress, Compression)
        
        

    def _clip_pre(self, aoi_fp=None, bbox=None, crs=None, clip_kwargs=None, **kwargs):
        #======================================================================= 
            # defaults 
            #=======================================================================
        log, tmp_dir, out_dir, ofp, resname= self._func_setup('clip', **kwargs)
        if clip_kwargs is None:
            clip_kwargs = dict()
        if not aoi_fp is None:
            self._set_aoi(aoi_fp)
        if bbox is None:
            bbox = self.bbox
        if crs is None:
            crs = self.crs
    #=======================================================================
    # update clipping kwargs
    #=======================================================================
        clip_kwargs.update(dict(bbox=bbox, crs=crs))
        return ofp, clip_kwargs, log, out_dir

    def clip_rlay(self, fp, ofp=None, **kwargs):
        """skinny clip a single raster
        

        """
        #=======================================================================
        # defaults
        #=======================================================================
        _, clip_kwargs, log, out_dir = self._clip_pre(**kwargs)
        
        if ofp is None: 
            fname = os.path.splitext( os.path.basename(fp))[0] + '_clip.tif'
            ofp = os.path.join(os.path.dirname(fp),fname)
            
        #=======================================================================
        # clip
        #=======================================================================
        ofp, stats_d = write_clip(fp, ofp=ofp, **clip_kwargs)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'clipped {os.path.basename(fp)}\n    {stats_d})')
        
        return ofp
        
    def clip_rlays(self, fp_d,
 
                  sfx='clip', 
                  **kwargs):
        """skinny clip a single raster
        
        Pars
        -----------
        aoi_fp: str
            filepath to aoi polygon.
            optional for setting excplicitly.
            aoi_fp passed to init is already set
        """
        ofp, clip_kwargs, log, out_dir = self._clip_pre(**kwargs)
            
        #=======================================================================
        # clip each
        #=======================================================================
        log.info(f'clipping {len(fp_d)} rasters to \n    %s'%clip_kwargs['bbox'])
        res_d = dict()
 
        for key, fp in fp_d.items(): 
            d={'og_fp':fp}
            d['clip_fp'], d['stats'] = write_clip(fp,ofp=os.path.join(out_dir, f'{key}_{sfx}.tif'),
                                                  **clip_kwargs)
            
            log.debug(f'clipped {key}:\n    {fp}\n    %s'%d['clip_fp'])
            
            res_d[key] = d
            
        log.info(f'finished on {len(res_d)}')
        return res_d
 
 
    
    def write_array(self,raw_ar,
                        #write kwargs
                       nodata=None, compress=None,
                       
                       #function kwargs
                       logger=None, out_dir=None, tmp_dir=None,ofp=None, 
                     resname=None,ext='.tif',subdir=False,
                       **kwargs):
        """write an array to raster using rio using session defaults
        
        TODO: use _get_defaults
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('write_array',
                           ext=ext, logger=logger, out_dir=out_dir, tmp_dir=tmp_dir, ofp=ofp,
                           resname=resname, subdir=subdir)
        
        if compress is None:
            compress = self.compress
        if nodata is None:
            nodata = self.nodata
        
        #=======================================================================
        # get kwargs
        #=======================================================================
        #pull the session defaults
        prof_d = self.get_profile()
        
        #update 
        prof_d.update({**dict(compress=compress, nodata=nodata), **kwargs})
 
        
        #=======================================================================
        # write
        #=======================================================================
        log.info(f'writing {str(raw_ar.shape)}  to \n    {ofp}')
 
        return write_array2(raw_ar, ofp, **prof_d)       
    
 
    
    def _get_defaults(self, crs=None, bbox=None, nodata=None, compress=None,
                      as_dict=False):
        """return session defaults for this worker
        
        EXAMPLE
        ----------
        crs, bbox, compress, nodata =RioSession._get_defaults(self)
        """
        self.assert_atts()
        
        if crs is None: crs=self.crs
        if bbox is  None: bbox=self.bbox
        if compress is None: compress=self.compress
        if nodata is None: nodata=self.nodata
        

        
        
        if not as_dict:
            return crs, bbox, compress, nodata
        else:
            return dict(crs=crs, bbox=bbox, compress=compress, nodata=nodata)
        
    def assert_atts(self):
        RioWrkr.assert_atts(self)
        SpatialBBOXWrkr.assert_atts(self)
        """not setup for this
        super().assert_atts()"""
 
 
        
        
class GridTypes(object):
    """handling and conversion of grid types"""
    fp=None
    
    def __init__(self,
             dkey, 
             fp=None,
             map_lib=None,
             conv_lib=None,
             out_dir=None,
             ):
        
        #=======================================================================
        # basics
        #=======================================================================
        self.dkey=dkey
        
        if not fp is None:
            assert os.path.exists(fp)
            self.fp=fp
            
        
        if out_dir is None:
            out_dir = tempfile.gettempdir()
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        self.out_dir=out_dir
        
        #=======================================================================
        # attachments
        #=======================================================================
        self.map_lib=map_lib
        self.conv_lib=conv_lib
        
        #=======================================================================
        # prechecks
        #=======================================================================
        if not self.fp is None:
            self.assert_fp()
            
    def assertd(self, *args, **kwargs):
        if not __debug__: # true if Python was not started with an -O option
            return 
        __tracebackhide__ = True
    
        return self.map_lib[self.dkey]['assert'](*args, **kwargs)
                         

    def assert_fp(self, fp=None, msg=''):
        """check the file matches the dkey expectations"""
        if not __debug__: # true if Python was not started with an -O option
            return 
        #__tracebackhide__ = True
        if fp is None:
            fp=self.fp
            
        dkey = self.dkey  
        
        #dkey check
        if not dkey in self.map_lib:
            raise AssertionError(f'unrecognized dkey {dkey}')
        
        #file type checking
        if not os.path.exists(fp):
            raise AssertionError(f'got bad filepath\n    {fp}\n'+msg)
        
        #apply the assertion

        assert_func =  self.map_lib[dkey]['assert']
        
        self.apply_fp(fp, assert_func, msg=msg+f' w/ dkey={dkey}')
        
    def apply_fp(self, fp, func, **kwargs):
        assert isinstance(fp, str)
        assert os.path.exists(fp)
        return self.map_lib[self.dkey]['apply'](fp, func, **kwargs)
    
    def load_fp(self, fp, **kwargs):
        #type check
        self.assert_fp(fp, msg='loading')
        
        #return the data
        return self.map_lib[self.dkey]['load'](fp, **kwargs)
    
    def convert(self, out_dkey,out_dir=None, **kwargs):
        """convert to the requested type"""
        #precheck
        if not self.dkey in self.conv_lib:
            raise KeyError(self.dkey)
        
        #defaults
        if out_dir is None: out_dir=self.out_dir
        
        #extract function
        d = self.conv_lib[self.dkey]
        
        if not out_dkey in d:
            raise NotImplementedError(out_dkey)
        
        f =d[out_dkey]
        
        #execute
        return f(out_dir=out_dir, **kwargs)
    
    def _get_ofp(self, sfx=None, ofp=None, out_dir=None, ext=None, 
                 fname=None, base_fp=None,):
        
        if ofp is None:
            #directory
            if out_dir is None:
                out_dir=self.out_dir
            
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            #extension
            if ext is None:
                ext = self.map_lib[self.dkey]['ext']
            
            
            #filename
            if fname is None:
                
                if base_fp is None:
                    base_fp=self.fp
                    
                fname = os.path.basename(base_fp).replace('.asc', '').replace(ext, '')
                
                if not sfx is None:
                    fname = fname+'_' + sfx
                    
            assert not ext in fname
            
            #assemble
            ofp = os.path.join(out_dir, fname+ext)
            
        return ofp
        
 
    

class ErrGridTypes(GridTypes):
    """type handling of error/confusion grids"""
    def __init__(self,dkey,
                 map_lib=None,
                 conv_lib=None, 
                 **kwargs):
        
        if map_lib is None:
            map_lib = dict()
            
        map_lib.update({ 
            'CONFU':        {},
             })
        
        super().__init__(dkey, map_lib=map_lib, conv_lib=conv_lib, **kwargs)
    
    
#===============================================================================
# HELPERS----------
#===============================================================================

 
def write_array2(ar, ofp, 
                 count=1, width=None, height=None, nodata=-9999, dtype=None,
                 transform=None, bbox=None,
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
    # bounded
    #===========================================================================
    if transform is None and bbox is not None:
        """otherwise result is mirrored about x=0?"""
        height, width  = ar.shape
        transform=rio.transform.from_bounds(*bbox.bounds,width, height)
    
    #===========================================================================
    # write
    #===========================================================================
    open_kwargs = {**dict(count=count, width=width, height=height,nodata=nodata, 
                       dtype=dtype,transform=transform, bbox=bbox), **kwargs}
    
    #print(dstr(open_kwargs))
    
    with rio.open(ofp, 'w',**open_kwargs) as ds:
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

    #print(f'writing {data.shape} to {ofp}')

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
        if not is_raster_file(rlay):
            raise AssertionError(f'expected a raster filepath:\n    {rlay}')
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
               buffer_bbox=False,
                round_offsets=False,
                 round_lengths=False,
                 ):
    """get a well rounded window from a bbox"""
    #buffer 1 pixel 
    if buffer_bbox: 
        bbox1 = sgeo.box(*bbox.buffer(ds.res[0], cap_style=3, resolution=1).bounds)
    else:
        bbox1 =bbox
    
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
    

def _get_meta(ds, att_l=['crs', 'height', 'width', 'transform', 'nodata', 'bounds', 'res', 'dtypes']):
    d = dict()
    for attn in att_l:        
        d[attn] = getattr(ds, attn)
    return d

def get_stats2(rlay, **kwargs):

    warnings.warn("deprecated (2023 01 28). use get_meta() instead", DeprecationWarning)
    return rlay_apply(rlay, lambda x:get_stats(x, **kwargs))

def get_meta(rlay, **kwargs):

    return rlay_apply(rlay, lambda x:_get_meta(x, **kwargs))

def get_ds_attr(rlay, stat):
    return rlay_apply(rlay, lambda ds:getattr(ds, stat))

def get_profile(rlay):
    return rlay_apply(rlay, lambda ds:ds.profile)

def get_crs(rlay): 
    #extract from metadata and convert to pyproj
    crs = CRS(get_ds_attr(rlay, 'crs'))
    assert isinstance(crs, CRS), f'bad type on crs from {rlay}\n    {type(crs)}'
    return crs

def get_write_kwargs( obj,
                      att_l = ['crs', 'transform', 'nodata'],
                      **kwargs):
    """convenience for getting write kwargsfrom datasource stats"""
    #=======================================================================
    # load from filepath
    #=======================================================================
    if isinstance(obj, str) or isinstance(obj,rasterio.io.DatasetReader):
        stats_d  = rlay_apply(obj, _get_meta, att_l=att_l+['dtypes'])
    elif isinstance(obj, dict):
        stats_d = obj

    else:
        raise TypeError(type(obj))
                        
    rlay_kwargs = {**kwargs,
        **{k:stats_d[k] for k in att_l}}  
          
    #handle tuple
    rlay_kwargs['dtype'] = stats_d['dtypes'][0]
    
    return rlay_kwargs

def get_shape(obj):
    d = rlay_apply(obj, _get_meta, att_l=['height', 'width'])
    
    return (d['height'], d['width'])

def get_support_ratio(obj_top, obj_bot):
        """get scale difference"""
        shape1 = get_shape(obj_top)
        shape2 = get_shape(obj_bot)
        
        height_ratio = shape1[0]/shape2[0]
        width_ratio = shape1[1]/shape2[1]
        
        assert height_ratio==width_ratio, f'ratio mismatch. height={height_ratio}. width={width_ratio}'
        
        return width_ratio


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

def get_bbox(rlay_obj):
    bounds = get_ds_attr(rlay_obj, 'bounds')
    return sgeo.box(*bounds)

def get_data_stats(fp, **kwargs):
    
    ar = load_array(fp, **kwargs)
    
    return {
        'max':ar.max(),
        'min':ar.min(),
        'mean':ar.mean(), #not sure how mask is treated
        'mask':ar.mask.sum(),
        'size':ar.size,
        }
    
 
    
def rlay_to_polygons(rlay_fp, convert_to_binary=True,
                          ):
    """
    get shapely polygons for each clump in a raster
    
    see also hp.hyd.write_inun_poly
    
    Parameters
    -----------
    convert_to_binary: bool, True
        True: polygon around mask values (e.g., inundation)
        False: polygon around data values
        
    """
    
    #===========================================================================
    # collect polygons
    #===========================================================================
    print(f'with convert_to_binary={convert_to_binary} on \n    {rlay_fp}')
    with rio.open(rlay_fp, mode='r') as src:
        mar = src.read(1, masked=True)
        
        if convert_to_binary:
            source = np.where(mar.mask, int(mar.fill_value),1)
        else:
            source = mar
        #mask = image != src.nodata
        d=dict()
        for geom, val in rasterio.features.shapes(source, mask=~mar.mask,
                                                  transform=src.transform,
                                                  connectivity=8):
            
            d[val] = sgeo.shape(geom)
            
        #print(f'finished w/ {len(d)} polygon')
        
 
        
    return d
    
#===============================================================================
# Building New Rasters--------
#===============================================================================
def copyr(fp, ofp):
    if ofp==fp:
        ofp=fp
    elif is_raster_file(fp):                    
        rasterio.shutil.copy(fp, ofp)
    else:
        shutil.copy(fp, ofp)
    return ofp
    
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
            dataset.read(1, masked=True)
            """
            assert not np.any(np.isnan(dataset.read(1, masked=False))), \
                'found masked nulls... resampling wont handle this?'
         
            out_shape=(dataset.count,int(dataset.height * scale),int(dataset.width * scale))
            print(out_shape)
 
            
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
 
            mar_raw = dataset.read_masks(1) #0=masked, 255=noMask
            
            if np.any(mar_raw==0):

                
                #downsample.disag. (zoom in)
                if scale>1.0:
                    #msk_rsmp = skimage.transform.resize(mar_raw, (out_shape[1], out_shape[2]), order=0, mode='constant')
                    msk_rsmp = scipy.ndimage.zoom(mar_raw, scale, order=0, mode='reflect',   grid_mode=True)
         
         
                #upsample. aggregate (those with ALL nulls)
                else:
                    """see also  hp.np.apply_block_reduce
                    
                    only works with downscales that are a power of 2?
                    """
                    #check shape
                    #assert np.all(np.array(out_shape[1:])%2==0), 'can only aggregate nulls with even shape?'
                    assert scale%1.0==0, 'only works for even scales'
     
                    #stack windows into axis 1 and 3
     
                    downscale = int(1/scale)
                    mar1 = mar_raw.reshape(mar_raw.shape[0]//downscale, 
                                           downscale, 
                                           mar_raw.shape[1]//downscale, 
                                           downscale)
                    
         
                    #those where the max of the children equals exactly the null value
                    msk_rsmp = np.where(np.max(mar1, axis=(1,3))==0, 0,255)
            else:
                """no nulls... allow more flexible aggregation (shape doesn't need to be divisible by 2)"""
                msk_rsmp = dataset.read_masks(1, 
                    out_shape=out_shape,
                    resampling=Resampling.nearest, #doesnt bleed
                    )
                
        
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
                
                fancy_window=None,
                
                ofp=None,
                
                **kwargs):
    """write a new raster from a window"""

    
    with rio.open(raw_fp, mode='r') as ds:
        
        #crs check/load
        if not crs is None:
            assert crs==ds.crs, f'mismatch crs={crs} (ds.crs={ds.crs}) on\n    {raw_fp}'
        else:
            crs = ds.crs
        
        #window default
        if window is None:
            """make the user pass window explicitly if they want rounding
            window, _ = get_window(ds, bbox, round_offsets=round_offsets, round_lengths=round_lengths)
            """
            if fancy_window is None:
                window = rasterio.windows.from_bounds(*bbox.bounds, transform=ds.transform)
            else: #get a well rounded window from a bbox
                window, _ = get_window(ds, bbox, **fancy_window)
 
        else: 
            assert bbox is None
            
        #get the windowed transform
        transform = rasterio.windows.transform(window, ds.transform)
        
        #get stats
        stats_d = _get_meta(ds)
        #stats_d['bounds'] = rio.windows.bounds(window, transform=transform)
            
        #load the windowed data
        ar = ds.read(1, window=window, masked=masked)
        
        """
        print(get_meta(ds))
        ds.read(1)
        """
        
        for e in ar.shape:
            assert e>0, window
 
        #=======================================================================
        # #write clipped data
        #=======================================================================
        if ofp is None:
            fname = os.path.splitext( os.path.basename(raw_fp))[0] + '_clip.tif'
            ofp = os.path.join(os.path.dirname(raw_fp),fname)
        
        write_kwargs = get_write_kwargs(ds)
        write_kwargs1 = {**write_kwargs, **dict(transform=transform), **kwargs}
        
        ofp = write_array2(ar, ofp,  masked=False,   **write_kwargs1)
        
    stats_d['bounds'] = get_ds_attr(ofp, 'bounds')
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
        stats_d = _get_meta(ds)
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




#===============================================================================
# PLOTS----------
#===============================================================================
def plot_with_window(ax, fp, bbox=None, **kwargs):
    """plot a clipped raster"""
    with rio.open(fp, mode='r') as ds:
        
        #===================================================================
        # #load and clip the array
        #===================================================================
        if bbox is None:
            window = None
            transform = ds.transform
        else:
            window = rio.windows.from_bounds(*bbox.bounds, transform=ds.transform)
            #transform = rio.transform.from_bounds(*bbox.bounds, *window.shape)
            transform = rio.windows.transform(window, ds.transform)
            
        ar = ds.read(1, window=window, masked=True)
    
    return show(ar, 
                transform=transform, 
                ax=ax, contour=False,interpolation='nearest', **kwargs)
 

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
    return ext in ['.tif', '.asc']


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
        raise AssertionError(f'non-square pixels {x} x {y}\n' + msg)
 
    if not round(x, 10)==int(x):
        raise AssertionError('non-integer pixel size\n' + msg)
    
def assert_extent_equal(left, right,  msg='',): 
    """ extents check"""
    if not __debug__: # true if Python was not started with an -O option
        return
 
    __tracebackhide__ = True
    
    f= lambda ds, att_l=['crs',  'bounds']:_get_meta(ds, att_l=att_l) 
    
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


def is_spatial_equal(left, right):
    f= lambda ds, att_l=['crs', 'height', 'width', 'bounds', 'res']:_get_meta(ds, att_l=att_l)
    
    ld = rlay_apply(left, f)
    rd = rlay_apply(right, f)
 
    for k, lval in ld.items():
        rval = rd[k]
        if not lval == rval:
            return False
        
    return True
            

def assert_spatial_equal(left, right,  msg='',): 
    """check all spatial attributes match"""
    if not __debug__: # true if Python was not started with an -O option
        return 
    __tracebackhide__ = True     
    
 
    f= lambda ds, att_l=['crs', 'height', 'width', 'bounds', 'res']:_get_meta(ds, att_l=att_l)
    
    ld = rlay_apply(left, f)
    rd = rlay_apply(right, f)
 
    for k, lval in ld.items():
        rval = rd[k]
        if not lval == rval:
            raise AssertionError(f'{k} mismatch\n    right={rval}\n    left={lval}\n' + msg)
        
 
        
        
        
     

def assert_ds_attribute_match(rlay,
                          crs=None, height=None, width=None, transform=None, nodata=None,bounds=None,
                          msg=''):

    #assertion setup
    if not __debug__: # true if Python was not started with an -O option
        return
    __tracebackhide__ = True
    
    stats_d = rlay_apply(rlay, _get_meta)
    
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
 

 


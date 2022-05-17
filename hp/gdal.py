'''
Created on Jul. 15, 2021

@author: cefect

gdal/ogr helpers

2021-07-24
    was getting some strange phantom crashing
    reverted and seems to be working again
'''


#===============================================================================
# imports----------
#===============================================================================
import time, sys, os, logging, copy, tempfile, datetime

from osgeo import ogr, gdal_array, gdal, osr

import numpy as np
import pandas as pd


from qgis.core import QgsVectorLayer, QgsMapLayerStore

from hp.exceptions import Error


mod_logger = logging.getLogger(__name__)

#===============================================================================
# classes------
#===============================================================================

class GeoDataBase(object): #wrapper for GDB functions
    
    def __init__(self,
                 fp, #filepath to gdb
                 name=None,
                 logger=mod_logger,
                 ):
        #=======================================================================
        # defaults
        #=======================================================================
        #logger setup
        self.logger = logger.getChild(os.path.basename(fp))
        if name is None: name = os.path.basename(fp)
        self.name=name
        self.fp = fp
        self.mstore = QgsMapLayerStore()
        #=======================================================================
        # #driver and data
        #=======================================================================
        self.driver = ogr.GetDriverByName("OpenFileGDB")
        
        self.data = self.driver.Open(fp, 0)
        
        #get layer info
        self.layerNames_l = [l.GetName() for l in self.data]
        
        logger.info('loaded GDB \'%s\' w/ %i layers \n    %s'%(
            name,len(self.layerNames_l), self.layerNames_l))
        
    def GetLayerByName(self, layerName, 
                       mstore=True, #whether to add the layer to the mstore (and kill on close)
                       ): #return a pyqgis layer
        
        assert layerName in self.layerNames_l, 'passed layerName not found \'%s\''%layerName
        
        #=======================================================================
        # load the layer
        #=======================================================================
        uri = "{0}|layername={1}".format(self.fp, layerName)
        
        vlay = QgsVectorLayer(uri,layerName,'ogr')
        
        #===========================================================================
        # checks
        #===========================================================================
        if not isinstance(vlay, QgsVectorLayer): 
            raise IOError
        
        #check if this is valid
        if not vlay.isValid():
            raise Error('loaded vlay \'%s\' is not valid. \n \n did you initilize?'%vlay.name())
        
        #check if it has geometry
        if vlay.wkbType() == 100:
            raise Error('loaded vlay has NoGeometry')
        
        #check coordinate system
        if not vlay.crs().isValid():
            raise Error('bad crs')
        
        if vlay.crs().authid() == '':
            print('\'%s\' has a bad crs'%layerName)
            
        #=======================================================================
        # wrap
        #=======================================================================
        
        if mstore: self.mstore.addMapLayer(vlay)
        
        self.logger.debug("loading with mstore=%s \n    %s"%(mstore, uri))
        
        return vlay
        
    def __enter__(self,):
        return self
    
    def __exit__(self, #destructor
                 *args,**kwargs):
        

        self.logger.debug('closing layerGDB')
        
        self.data.Release()
        
        self.mstore.removeAllMapLayers()
        
        #super().__exit__(*args,**kwargs) #initilzie teh baseclass
        

#===============================================================================
# functions------
#===============================================================================
def get_layer_gdb_dir( #extract a specific layer from all gdbs in a directory
                 gdb_dir,
                 layerName='NHN_HD_WATERBODY_2', #layername to extract from GDBs
                 logger=mod_logger,
                 ):
    
    #=======================================================================
    # defaults
    #=======================================================================
    log=logger.getChild('get_layer_gdb_dir')
    
    
    #=======================================================================
    # #get filepaths
    #=======================================================================
    fp_l = [os.path.join(gdb_dir, e) for e in os.listdir(gdb_dir) if e.endswith('.gdb')]
    
    log.info('pulling from %i gdbs found in \n    %s'%(len(fp_l), gdb_dir))
    
    #=======================================================================
    # load and save each layer
    #=======================================================================
    d = dict()
    for fp in fp_l:

        """need to query layers in the gdb.. then extract a specific layer"""
         
        
        with GeoDataBase(fp, logger=log) as db:
            assert not db.name in d, db.name
            d[copy.copy(db.name)] = db.GetLayerByName(layerName, mstore=False)
        
 
            
    
    log.info('loaded %i layer \'%s\' from GDBs'%(len(d), layerName))
    
    return d

def get_nodata_val(rlay_fp):
    assert os.path.exists(rlay_fp)
    ds = gdal.Open(rlay_fp)
    band = ds.GetRasterBand(1)
    return band.GetNoDataValue()
    
    


def rlay_to_array(rlay_fp, dtype=np.dtype('float32')):
    """context managger?"""
    #get raw data
    ds = gdal.Open(rlay_fp)
    band = ds.GetRasterBand(1)
    
    
    ar_raw = np.array(band.ReadAsArray(), dtype=dtype)
    
    #remove nodata values
    ndval = band.GetNoDataValue()
    
    ar_raw[ar_raw==ndval]=np.nan
    
    del ds
    del band
    
    return ar_raw

def array_to_rlay(ar_raw, #convert a numpy array to a raster
                  
                  #raster properties
                  pixelWidth=None,
                  pixelHeight=None,
                  resolution=1.0,
                  rasterOrigin=(0,0), #upper left
                  epsg=4326,
                  nodata=-9999,
                  
                  #outputs
                  layname='array_to_rlay',
                  ofp=None,
                  out_dir=None,
                  ):
    """adapted from here:
    https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#create-raster-from-array
    """

    #===========================================================================
    # defaults
    #===========================================================================
    if pixelWidth is None and pixelHeight is None:
        assert isinstance(resolution, float)
        pixelWidth=resolution
        pixelHeight=resolution
        
    if out_dir is None:
        out_dir = os.path.join(tempfile.gettempdir(), __name__, 
                               datetime.datetime.now().strftime('%Y%m%d%M%S'))
        if not os.path.exists(out_dir): 
            os.makedirs(out_dir)
        
    if ofp is None:
        ofp = os.path.join(out_dir,layname+'.tif')
    
    assert isinstance(ar_raw, np.ndarray)
    #===========================================================================
    # extract
    #===========================================================================
    assert len(ar_raw.shape)==2, ar_raw.shape
    cols = ar_raw.shape[1]
    rows = ar_raw.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]
    
    #===========================================================================
    # clean nodat
    #===========================================================================
    array = ar_raw.copy()
    array[np.isnan(array)]=nodata
    #===========================================================================
    # construct
    #===========================================================================
    #===========================================================================
    driver = gdal.GetDriverByName('GTiff')
    # outRaster = driver.Create(ofp, cols, rows, 1, gdal.GDT_Byte)
    # outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    # outband = outRaster.GetRasterBand(1)
    # outband.WriteArray(array)
    #===========================================================================
    # outRasterSRS = osr.SpatialReference()
    # outRasterSRS.ImportFromEPSG(epsg)
    #===========================================================================
    # outRaster.SetProjection(outRasterSRS.ExportToWkt())
    # outband.FlushCache()
    # outRaster=None #close
    #===========================================================================
    
    dst_ds = driver.Create(ofp, xsize=cols, ysize=rows,
                    bands=1, eType=gdal.GDT_Float32)
    
    #dst_ds.SetGeoTransform([444720, 30, 0, 3751320, 0, -30])
    dst_ds.SetGeoTransform([originX, pixelWidth, 0, originY, 0, -pixelHeight])
    
    #build spatial ref
    #===========================================================================
    # srs = osr.SpatialReference()
    # srs.SetUTM(11, 1)
    # srs.SetWellKnownGeogCS("NAD27")
    #===========================================================================
    
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    
    dst_ds.SetProjection(outRasterSRS.ExportToWkt())
 
    band = dst_ds.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    # Once we're done, close properly the dataset
    dst_ds = None
    
    return ofp

def getRasterMetadata(fp):
    assert os.path.exists(fp)
    
    dataset = gdal.OpenEx(fp)
    
    md = copy.copy(dataset.GetMetadata('IMAGE_STRUCTURE'))
    
    del dataset
    
    return md
    
def getRasterCompression(fp):
    md = getRasterMetadata(fp)
    
    if not 'COMPRESSION' in md:
        return None
    else:
        return md['COMPRESSION']   
    
def getRasterStatistics(fp):
    ds = gdal.Open(fp)
 
    band = ds.GetRasterBand(1)
    d = dict()
    d['min'], d['max'], d['mean'], d['stddev'] = band.GetStatistics(True, True)
 
    
    del ds
    
    return d

def getNoDataCount(fp, dtype=np.dtype('float')):
    """2022-05-10: this was returning some nulls
    for rasters where I could not find any nulls"""
    #get raw data
    ds = gdal.Open(fp)
    band = ds.GetRasterBand(1)
    
    
    ar_raw = np.array(band.ReadAsArray(), dtype=dtype)
    
    #remove nodata values
    ndval = band.GetNoDataValue()
    
    #get count
    bx_ar = ar_raw == ndval
    
    del ds
    del band
 
    return bx_ar.sum()
    
 
    
    
                
            
if __name__ =="__main__": 
    rlay_fp = r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\wd\present\wd_1grid.tif'
    
    ar = rlay_to_array(rlay_fp)
    
    import pandas as pd
    df = pd.DataFrame(ar)
    
    df.to_csv(r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\wd\present\wd_1grid.csv')
           
            
            
            
            
            
            
            
            
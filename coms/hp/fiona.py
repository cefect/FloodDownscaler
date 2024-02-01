'''
Created on Mar. 8, 2023

@author: cefect
'''
import os
import fiona #not a rasterio dependency? needed for aoi work
from pyproj.crs import CRS
import shapely.geometry as sgeo
from shapely.geometry import mapping
from shapely.geometry.polygon import Polygon
from rasterio.crs import CRS as CRS_rasterio 


def get_bbox_and_crs(fp):
    with fiona.open(fp, "r") as source:
        bbox = sgeo.box(*source.bounds) 
        crs = CRS(source.crs['init'])
        
    return bbox, crs


def write_bbox_vlay(bbox, crs, ofp):

    #write a vectorlayer from a single bounding box
    assert isinstance(bbox, Polygon)
    with fiona.open(ofp,'w',driver='GeoJSON', 
        crs=fiona.crs.from_epsg(crs.to_epsg()),
        schema={'geometry': 'Polygon',
                'properties': {'id':'int'},
            },
 
        ) as c:
        
        c.write({ 
            'geometry':mapping(bbox), 
            'properties':{'id':0},
            })
        
    return ofp


class SpatialBBOXWrkr(object):
    aoi_fp=None
    crs=None
    bbox=None
    
    def __init__(self, 
                 #==============================================================
                 # crs=CRS.from_user_input(25832),
                 # bbox=
                 #==============================================================
                 crs=None, bbox=None, aoi_fp=None,
                 
                 #defaults
                 init_pars=None,
                 
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
        if init_pars is None: init_pars=list()
        
        #=======================================================================
        # set aoi
        #=======================================================================
        if not aoi_fp is None:
                        
            assert bbox is None
            self._set_aoi(aoi_fp)
            init_pars.append('aoi_fp')
            
            if not crs is None:
                assert crs==self.crs, f'crs mismatch between aoi {self.crs} and data {crs}'
 
        else:
            self.crs=crs
            self.bbox = bbox
            
        
            
        #check
        if not self.crs is None:
            assert isinstance(self.crs, CRS)
            init_pars.append('crs')
            
        if not self.bbox is None:
            assert isinstance(self.bbox, Polygon)
            init_pars.append('bbox')
        
        
        super().__init__(init_pars=init_pars, **kwargs)
        
        if not aoi_fp is None:
            self.logger.info(f'set crs:{self.crs.to_epsg()} from {os.path.basename(aoi_fp)}')
            
         
  
    def _set_aoi(self, aoi_fp):
        assert os.path.exists(aoi_fp)
        
        #open file and get bounds and crs using fiona
        bbox, crs = get_bbox_and_crs(aoi_fp) 
            
        self.crs=crs
        self.bbox = bbox        
        self.aoi_fp=aoi_fp
        
        return self.crs, self.bbox
    
    def assert_atts(self):
        #check
        if not self.bbox is None: #optionakl
            assert isinstance(self.bbox, sgeo.Polygon), f'bad bbox type: {type(self.bbox)}'
            assert hasattr(self.bbox, 'bounds')
            
        assert isinstance(self.crs, CRS) or isinstance(self.crs, CRS_rasterio), f'bad type on crs ({type(self.crs)})'
        
    def write_bbox_vlay(self,bbox,crs=None,
                        **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('bbox', ext='.geojson', **kwargs)
        if crs is None: crs=self.crs
        
        return write_bbox_vlay(bbox, crs, ofp)
        
        
 
        
        
        
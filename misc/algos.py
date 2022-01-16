'''
Created on Mar. 27, 2021

@author: cefect

running simple algos
'''
#===============================================================================
# imports-----------
#===============================================================================
import os, datetime


start =  datetime.datetime.now()
print('start at %s'%start)
today_str = datetime.datetime.today().strftime('%Y%m%d')


from hp.exceptions import Error
from hp.dirz import force_open_dir
from hp.oop import Basic
from hp.plot import Plotr #only needed for plotting sessions
from hp.Q import Qproj #only for Qgis sessions
 


#===============================================================================
# vars
#===============================================================================
work_dir = r'C:\LS\03_TOOLS\misc'


#===============================================================================
# CLASSES----------
#===============================================================================
        
        
class SessionQ(Qproj):
    
    def __init__(self,
                 out_dir = os.path.join(work_dir, 'algos', today_str),
                 crsID_default = 'EPSG:3857',
                 ):
        
        super().__init__(work_dir=r'C:\LS\03_TOOLS\misc',
                         out_dir=out_dir, crsID_default=crsID_default)
        






#===============================================================================
# FUNCTIONS-----------------
#===============================================================================

def dissolve_wAoi(
        aoi_fp = r'C:\LS\02_WORK\02_Mscripts\InsuranceCurves\04_CALC\QC\aoi\aoi_HRDEM_QC_0506.gpkg',
        vlay_fp = r'C:\LS\03_TOOLS\misc\py\algos\20210506\EGS_Flood_Product_Archive_20210329_aoi_fixd.gpkg'
        ):
    
    """
    still too slow. could try:
        multi to single part
            dropping small single parts
        simplify
        smaller aoi
        
    """
    
    wrkr = SessionQ()
    log = wrkr.logger
    
    aoi_vlay = wrkr.vlay_load(aoi_fp)
    
    vlay = wrkr.vlay_load(vlay_fp)
    
    #===========================================================================
    # #select
    # wrkr.selectbylocation(vlay, aoi_vlay)
    # 
    # #fix geo
    # log.info('fixing geo on %i feats'%vlay.dataProvider().featureCount())
    # ofn = '%s_aoi_fixd.gpkg'%vlay.name()
    # vlay_fixd = wrkr.fixgeo(vlay, selected_only=True, output=os.path.join(wrkr.out_dir, ofn))
    #===========================================================================
    vlay_fixd = vlay
 

    log.info('disolving on %s'%vlay_fixd)
    ofn = '%s_aoi_disl.gpkg'%vlay.name()
    wrkr.dissolve(vlay_fixd, output=os.path.join(wrkr.out_dir, ofn))
    
    return wrkr.out_dir

def pointonsurf(
        fp = r'C:\LS\05_DATA\Global\Microsoft\CanadianBuildingFootprints\20210506_ON\Ontario.geojson'):
    wrkr = SessionQ()
    log = wrkr.logger
    vlay = wrkr.vlay_load(fp, addSpatialIndex=False)
    log.info('getting points for %i feats'%vlay.dataProvider().featureCount())
    wrkr.pointonsurf(vlay, output=os.path.join(wrkr.out_dir, '%s_pos.gpkg'%vlay.name()))
    
    return wrkr.out_dir



if __name__ =="__main__": 
    
    #out_dir = pointonsurf()
    out_dir = dissolve_wAoi()
    
    force_open_dir(out_dir)
    tdelta = datetime.datetime.now() - start
    print('finished in %s'%tdelta)

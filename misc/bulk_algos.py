'''
Created on Feb. 25, 2022

@author: cefect

misc scripts for executing algos on a directory of files
'''
import os, shutil


from qgis.core import QgsCoordinateReferenceSystem, QgsVectorLayer, QgsMapLayerStore, \
    QgsWkbTypes

from hp.Q import Qproj, view
from hp.exceptions import Error

def get_fps(search_dir, ext):
    fps_all = set()
    for dirpath, _, fns in os.walk(search_dir):
        fps_all.update([os.path.join(dirpath, e) for e in fns if e.endswith(ext)])
        
    return fps_all


def selectbylocation(
        left_fp = r'C:\LS\05_DATA\Global\Microsoft\CanadianBuildingFootprints\20210407_NB\NewBrunswick.geojson',
        right_fp = r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\SaintJohn\aoi13_0116.gpkg',
        ofp=None,
        ):
    
    with Qproj() as ses:
        left_vlay = ses.vlay_load(left_fp, set_proj_crs=True)
        ses.selectbylocation(left_vlay, right_fp)
        
        if ofp is None: ofp = os.path.join(ses.out_dir, 'selected.gpkg')
        
        ses.saveselectedfeatures(left_vlay, output=ofp) 
        
        print('saved to %s'%ofp)

def algo_byName(fps, #filepaths to execute algo on
                algoName='fillnodata',
                  out_dir=os.getcwd(),
                      sfx='fnd', ext = '.tif',
                      overwrite=False,
                      compression='med',
                      preload=True,
                      algoKwargs={},
                      **kwargs):
    
    print('\'%s\' on %i'%(algoName, len(fps)))
    res_d = dict()
    with Qproj(overwrite=overwrite, compression=compression, out_dir=out_dir, **kwargs) as ses:
        
        for i, fp in enumerate(fps):
            mstore = QgsMapLayerStore()
            log = ses.logger.getChild(str(i)) 
            print('on %i/%i:    %s'%(i+1, len(fps), fp))
            fname = os.path.basename(fp).replace(ext, '') + sfx
            
            #===================================================================
            # load the layer
            #===================================================================
            if preload:
                """better to load to get consistent algo handling"""
                if ext =='.tif':
                    raw_lay = ses.rlay_load(fp, set_proj_crs=True, logger=log)
                    mstore.addMapLayer(raw_lay)
                else:
                    raise Error('not implemented')
                
            else:
                raw_lay = fp
            
            #===================================================================
            # #execute the algo
            #===================================================================
            print('    calling %s'%algoName)
            f = getattr(ses, algoName)            
            res_lay = f(raw_lay, logger=log, **algoKwargs)
            
            #handle outputs
            """some algos always spit out fps"""
            if isinstance(res_lay, str):
                if res_lay.endswith('.tif'):
                    res_lay = ses.rlay_load(res_lay, set_proj_crs=False, logger=log)
                    mstore.addMapLayer(res_lay)
                else:
                    raise Error('not im[plemented')
                    
            
 
            #===================================================================
            # #save with some compression
            #===================================================================
            ofp = os.path.join(out_dir, fname + ext)
            
            if ext =='.tif':
                res_d[fname] = ses.rlay_write(res_lay, ofp=ofp, logger=log)
            else:
                raise Error('not implemented')
            
            #===================================================================
            # wrap
            #===================================================================
            mstore.removeAllMapLayers()
            print('    finished on \'%s\''%fname)
            
    print('finished on %i'%len(res_d))
    
    return res_d

def fillnodata_simple(
        fps, algoKwargs={},
        **kwargs):
    return algo_byName(fps, algoName='fillnodata', algoKwargs=algoKwargs, **kwargs) 
            
            

if __name__ == '__main__':
    
    
    fps = get_fps(
        r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wd\0116',
        #r'C:\LS\10_OUT\2112_Agg\ins\hyd\LMFRA\wd\0116',
        '.tif'
        )
    fillnodata_simple(fps,
                      compression='topo_lo',
                      out_dir = r'C:\LS\10_OUT\2112_Agg\ins\hyd\Calgary\wd\0116_fnd',
                      algoKwargs = dict(fval = 0),
                      )
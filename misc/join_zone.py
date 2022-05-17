'''
Created on Apr. 21, 2021

@author: cefect

add a 'zone' attribute from some polygons to a set of vector layers
'''


#===============================================================================
# imports
#===============================================================================
import os, datetime
import pandas as pd
import numpy as np
from hp.Q import Qproj, view, vlay_get_fdf, vlay_get_geo, predicate_d

from hp.dirz import force_open_dir
import processing 


wrkr = Qproj(crsID_default='EPSG:3005',
             out_dir=r'C:\LS\02_WORK\NHC\202103_NewWest\04_CALC\cf\s1\finv_z')


def collect_fps( #get files of interest
        data_dir=None,
        ):
    
    log = wrkr.logger.getChild('collect_fps')
    
    #===========================================================================
    # collect data filesnames
    #===========================================================================
    fp_d = dict()
    for dirpath, dirnames, filenames in os.walk(data_dir):
        if not os.path.basename(dirpath)=='res_djoin': #only want these
            continue
        
        mName = os.path.basename(os.path.dirname(dirpath))
        
        #special shortcut for ags
        """made an extra file here"""
        if mName=='ag.crops':
            fp_d[mName] = r'C:\LS\03_TOOLS\CanFlood\outs\lang\0224\s1\assetModels\ag.crops\res_djoin\r_passet_s1_ag.crops.gpkg'
        else:
            vfnames = [e for e in filenames if e.endswith('.gpkg')]
            assert len(vfnames)==1, '%s got too many files: \n    %s'%(mName, vfnames)
            fp_d[mName] = os.path.join(dirpath, vfnames[0])
            
    log.info('collected %i files: \n    %s'%(len(fp_d), fp_d.keys()))
    
    return fp_d



def join_zone(fp_d,
              zone_fp = None,

                          
                  ): #join zone value and collect data
    
    #===========================================================================
    # defaults
    #===========================================================================
    log = wrkr.logger.getChild('join_zone')
    
    #===========================================================================
    # check
    #===========================================================================
    assert isinstance(fp_d, dict)
    #===========================================================================
    # load the zonal
    #===========================================================================
    zvlay= wrkr.vlay_load(zone_fp)
    """
    view(zvlay)
    """
    
    log.info('on %i'%len(fp_d))
    

    
    #===========================================================================
    # join zone and collect results
    #===========================================================================
    res_d = dict()
    for mName, fp in fp_d.items():
        log = wrkr.logger.getChild(mName)
        assert os.path.exists(fp)
        assert fp.endswith('.gpkg')
        
        #load the data
        vlay = wrkr.vlay_load(fp)
        wrkr.mstore.addMapLayer(vlay)
        
        #output 
        fn = '%s_zd.gpkg'%mName.replace('_aoi', '')
        ofp = os.path.join(wrkr.out_dir, fn)
        
        algo_nm = 'qgis:joinattributesbylocation'
        
        pars_d = { 'DISCARD_NONMATCHING' : True, 
                  'INPUT' : vlay, 
                  'JOIN' : zvlay, 
                  'JOIN_FIELDS' : ['zone', 'zone_id'], 
                  'METHOD' : 0, 
                  'NON_MATCHING' : 'TEMPORARY_OUTPUT', 
                  'OUTPUT' : ofp, 
                  'PREDICATE' : [predicate_d['intersects']], #only accepting single predicate
                  'PREFIX' : '' }
        
        d = processing.run(algo_nm, pars_d, feedback=wrkr.feedback)
        
        #=======================================================================
        # #meta
        #=======================================================================
        nm_cnt = d['NON_MATCHING'].dataProvider().featureCount()
        d['miss_cnt'] = nm_cnt
        d['og_cnt'] = vlay.dataProvider().featureCount()
        d['fn'] = fn
        
        del d['NON_MATCHING']
        
        
        
        
        res_d[mName] = d
        
        #=======================================================================
        # wrap
        #=======================================================================
        wrkr.mstore.removeAllMapLayers() #clear the store


    #===========================================================================
    # wrap
    #===========================================================================
    log = wrkr.logger.getChild('join_zone')
    log.info('finished on %i'%len(res_d))
    
    return res_d
    
    







if __name__ =="__main__": 
    start =  datetime.datetime.now()
    print('start at %s'%start)
    
    
    
    #get file paths
    #fp_d = collect_fps(data_dir=r'C:\LS\02_WORK\NHC\202103_NewWest\04_CALC\cf\s1\finv')
    data_dir = r'C:\LS\02_WORK\NHC\202103_NewWest\04_CALC\cf\s1\finv'
    fn_l = [e for e in os.listdir(data_dir) if e.endswith('.gpkg')]
    fp_d = {os.path.splitext(e)[0]:os.path.join(data_dir, e) for e in fn_l}
    
    #join zone
    
    res_d = join_zone(fp_d,
                   zone_fp = r'C:\LS\02_WORK\NHC\202103_NewWest\04_CALC\aoi\aoi03_NewWest_20210421.gpkg')
    
    
    
    #write results
    ofp = os.path.join(wrkr.out_dir, 'jZone_meta_%s.csv'%datetime.datetime.now().strftime('%Y%m%d'))
    pd.DataFrame.from_dict(res_d, orient='index').to_csv(ofp)
    
    #===========================================================================
    # wrap
    #===========================================================================
    
    #force_open_dir(wrkr.out_dir)
    
    
    
    
    
    print('finished in %s'%(datetime.datetime.now() - start))
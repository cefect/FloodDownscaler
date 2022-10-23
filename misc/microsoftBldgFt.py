'''
Created on Mar. 27, 2021

@author: cefect

helpers for working with Microsoft Building Footprints
'''
#===============================================================================
# imports-----------
#===============================================================================
import os, datetime
import pandas as pd
import numpy as np

start =  datetime.datetime.now()
print('start at %s'%start)
today_str = datetime.datetime.today().strftime('%Y%m%d')


from hp.exceptions import Error
from hp.dirz import force_open_dir
from hp.oop import Basic
from hp.plot import Plotr #only needed for plotting sessions
from hp.Q import Qproj, processing #only for Qgis sessions
 


#===============================================================================
# vars
#===============================================================================



#===============================================================================
# CLASSES----------
#===============================================================================

        
        
class mbfpSession(Qproj):
    """retrieing MicrosoftBuildingFootprint points"""
    
    data_dir=r'C:\LS\05_DATA\Global\Microsoft\CanadianBuildingFootprints'
    aoi_vlay = None
    raw_lib=dict()
        
    def __init__(self,
                 aoi_fp = None,
                  tag='mbfp',
                  work_dir = r'C:\LS\10_IO\coms',
                 **kwargs):
        
        super().__init__(
                        tag=tag, work_dir=work_dir,
                         **kwargs)  # initilzie teh baseclass
        
        assert os.path.exists(self.data_dir)
        
        if not aoi_fp is None:
            #get the aoi
            aoi_vlay = self.load_aoi(aoi_fp, set_proj_crs=True)
            
    def build_geojson_library(self,
                       data_dir = None):
 
        """build reference to all the raw geojsons"""
        #=======================================================================
        # defaults
        #=======================================================================
        if data_dir is None: data_dir = self.data_dir
        log=self.logger.getChild('get_microsoft_fps')
        
        first = True
 
        d = dict()
        for dirpath, dirnames, filenames in os.walk(data_dir):
            if first: #skip the main
                first=False
                continue
            
            #search all the files to get the 'prov' key
            for fp in filenames:
                filename, ext = os.path.splitext(fp)
                if ext=='.geojson':
                    assert not filename.lower() in d, filename.lower()
                    d[filename.lower()] = os.path.join(dirpath, fp)
 
            
        log.info('got %i \n%s'%(len(d), d))
        #for k,v in d.items():print('%s        %s'%(k,v))
        
        return d
            
            
    def get_microsoft_fps(self,  #get geopackage filepaths by province
                       data_dir = None):
        """loading pre-made poinst"""
        
        #=======================================================================
        # defaults
        #=======================================================================
        if data_dir is None: data_dir = self.data_dir
        log=self.logger.getChild('get_microsoft_fps')
        
        first = True
        prov=None
        d = dict()
        for dirpath, dirnames, filenames in os.walk(data_dir):
            if first: #skip the main
                first=False
                continue
            
            log.debug('%s    %s    %s' % (dirpath, dirnames, filenames))
            
            #search all the files to get the 'prov' key
            for fp in filenames:
                filename, ext = os.path.splitext(fp)
                if ext=='.geojson':
                    prov = filename.lower()
                    break
            assert isinstance(prov, str), 'failed to get province on %s'%dirpath
                    
            #get data
            fn_d = {fp:os.path.splitext(fp)[1] for fp in filenames}
            if '.gpkg' in fn_d.values():
                sel_fp = {v:k for k,v in fn_d.items()}['.gpkg']
            else:
                sel_fp= fp #take from key loop

            assert not prov in d, prov
            d[prov]=os.path.join(dirpath, sel_fp)
            prov=None
            
        log.info('got %i \n%s'%(len(d), d))
        #for k,v in d.items():print('%s        %s'%(k,v))
        
        return d
    
 
    
    def build_pts(self, #get points for the microsoft data 
                          fp_d):
        """
        just getting fast/sloppy points on surface (skipping invalid geometry)
        """
        
        log = self.logger.getChild('get_microsoft_pts')
        res_d = dict()
        for prov, fp in fp_d.items():
            log =self.logger.getChild('gmp_%s'%prov)
            fdir, fn = os.path.split(fp)
            fn_str, ext = os.path.splitext(fn)
            if 'pts' in fn:
                log.debug('already points... skipping')
                res_d[prov] = fp
                continue
            
            #===================================================================
            # convert
            #===================================================================
            start =  datetime.datetime.now()

            algo_nm = 'native:pointonsurface'  
            ins_d = { 'ALL_PARTS' : False,'INPUT' : fp,'OUTPUT' : 'TEMPORARY_OUTPUT'}
                    
            log.info('executing \'%s\' on \'%s\' with: \n     %s'
                      %(algo_nm, prov, ins_d))
                    
            vlay = processing.run(algo_nm, ins_d,  feedback=self.feedback, context=self.context,
                                  )['OUTPUT']
            
            layname = 'microsoft_CanBldgFt_%s_%s_pts'%(prov, self.today_str)
            vlay.setName(layname)
            
            #===================================================================
            # save
            #===================================================================

            ofp = os.path.join(fdir, vlay.name()+'.gpkg')
            log.info('finished in %s \n    write %s to %s'%(datetime.datetime.now() - start, prov, ofp))
            self.vlay_write(vlay, ofp, logger=log)
            
            #===================================================================
            # wrap
            #===================================================================
            self.mstore.addMapLayer(vlay)
            self.mstore.removeAllMapLayers()
            res_d[prov] = ofp
            
            
            
            
        log.info('finished on %i'%len(res_d))
        
        return res_d
    
    def get_selection(self, #load a set of layers, slice by aoi, report on feature counts
                main_fp, 
                aoi_vlay = None,
                ofp = None,
                pred_l = ['intersect'],  #list of geometry predicate names
                logger=None,
                  ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        if aoi_vlay is  None: aoi_vlay=self.aoi_vlay
        log=logger.getChild('get_selection')
        
        
        #=======================================================================
        # load
        #=======================================================================3
        """need to load to hold selection?"""
        log.info('loading %s'%os.path.basename(main_fp))
        vlay = self.vlay_load(main_fp, logger=log, reproj=False)
        
 
        self.mstore.addMapLayer(vlay)
        #=======================================================================
        # reproject the aoi
        #=======================================================================
        aoi_reproj = self.reproject(aoi_vlay, crsOut=vlay.crs(), logger=log)

        #=======================================================================
        # make selection
        #=======================================================================
        algo_nm = 'native:selectbylocation'   
        pred_d = {
                'are within':6,
                'intersect':0,
                'overlap':5,
                  }
                
                
        ins_d = { 
            'INPUT' : vlay, 
            'INTERSECT' : aoi_reproj, 
            'METHOD' : 0, #new selection 
            'PREDICATE' : [pred_d[pred_nm] for pred_nm in pred_l]}
        
        log.info('executing \'%s\'   with: \n     %s'
            %(algo_nm,  ins_d))
            
        _ = processing.run(algo_nm, ins_d,  feedback=self.feedback)
        
        #=======================================================================
        # save selected and reproject
        #=======================================================================
        log.info('selected %i (of %i) features from %s'
            %(vlay.selectedFeatureCount(),vlay.dataProvider().featureCount(), vlay.name()))
        
        """doing this incase the repeojct fails?"""
        sel_fp = processing.run('native:saveselectedfeatures', 
                                  {'INPUT' : vlay,'OUTPUT' :os.path.join(self.temp_dir, '%s_sel.gpkg'%vlay.name())},  
                                  feedback=self.feedback)['OUTPUT']
        #self.mstore.addMapLayer(vlay_sel)
        log.info(sel_fp)
 
        #=======================================================================
        # reproject
        #=======================================================================
        if ofp is None: ofp = os.path.join(self.out_dir, '%s_sel.gpkg'%vlay.name())
        self.reproject(sel_fp, output=ofp, logger=log, selected_only=False, crsIn=vlay.crs())
         
        log.info('finished w/ %s'%ofp)
         
         
         
        return ofp
    

    
    


#===============================================================================
# FUNCTIONS-----------------
#===============================================================================

def get_mbfp( #get footprints
        aoi_fp=r'C:\LS\02_WORK\02_Mscripts\InsuranceCurves\04_CALC\CMM\aoi\aoi02_CMM_20210711.gpkg',
        prov='quebec', #province
        points=True,
        ):
    
    with mbfpSession(aoi_fp=aoi_fp) as wrkr:
        
        
        if points:
            
            raw_lib = wrkr.get_microsoft_fps()
 
            
            
            
        else:
            raw_lib = wrkr.build_geojson_library()
 
 
            
        ofp = wrkr.get_selection(raw_lib[prov])
        
    return ofp

 
    
    
def Fred12():
    
    return get_pts(
        aoi_fp=r'C:\LS\02_WORK\02_Mscripts\InsuranceCurves\04_CALC\Fred\aoi\aoi12_fred_0722.gpkg',
        prov='NewBrunswick'.lower()
        )

def obwb_0715():
    return get_mbfp(
        aoi_fp=r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\LMFRA\aoi\aoi10_20220715.gpkg',
        prov='BritishColumbia'.lower(), points=False
        )


def calgary_0715():
    return get_mbfp(
        aoi_fp=r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\Calgary\aoi\aoi03_20220715.gpkg',
        prov='Alberta'.lower(), points=False
        )

def sj_0819():
    return get_mbfp(
        aoi_fp=r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\SaintJohn\aoi\aoi17_20220819.geojson',
        prov='NewBrunswick'.lower(), points=False
        )








if __name__ =="__main__": 
    
    #obwb_0715()
    sj_0819()

    
    tdelta = datetime.datetime.now() - start
    print('finished in %s'%tdelta)

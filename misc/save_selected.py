'''
Created on Dec. 23, 2020

@author: cefect

save a set of features where overlapping with some other layer
'''
#===============================================================================
# script paths
#===============================================================================

#===============================================================================
# # standard imports -----------------------------------------------------------
#===============================================================================
import os, datetime
start =  datetime.datetime.now()
print('start at %s'%start)
    
from hp.dirz import force_open_dir
    
import processing 

#===============================================================================
# custom imports
#===============================================================================

from hp.Q import Qproj, view, vlay_get_fdf, QgsFeatureRequest
from hp.exceptions import Error


class Sliceor(Qproj):
    
    aoi_vlay = None
        
    def __init__(self,
                 aoi_fp = None,
                 **kwargs):
        
        super().__init__(

                         **kwargs)  # initilzie teh baseclass
        
        if not aoi_fp is None:
            #get the aoi
            aoi_vlay = self.load_aoi(aoi_fp)
        
    def run_slice(self, #load a set of layers, slice by aoi, report on feature counts
                main_fp, top_fp,
                ofp = None,
                pred_l = ['intersect'],  #list of geometry predicate names
                logger=None,
                write_csv=True,
                  ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('run_slice')
        
        #=======================================================================
        # load
        #=======================================================================3
        """need to load to hold selection?"""
        vlay = self.vlay_load(main_fp)
        
        if ofp is None: ofp = os.path.join(self.out_dir, '%s_sel.gpkg'%vlay.name())
        
        #=======================================================================
        # fix geo
        #=======================================================================
        log.info('fixing geometrty on %s'%top_fp)
        top_vlay = processing.run('native:fixgeometries', 
                                  {'INPUT' : top_fp,'OUTPUT' : 'TEMPORARY_OUTPUT'},
                                   feedback=self.feedback)['OUTPUT']
                                   
        top_vlay.setName(os.path.basename(top_fp).replace('.gpkg', ''))
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
            'INTERSECT' : top_vlay, 
            'METHOD' : 0, #new selection 
            'PREDICATE' : [pred_d[pred_nm] for pred_nm in pred_l]}
        
        log.info('executing \'%s\'   with: \n     %s'
            %(algo_nm,  ins_d))
 
        # #execute
 
        _ = processing.run(algo_nm, ins_d,  feedback=self.feedback)
        
        #=======================================================================
        # save selected
        #=======================================================================
        log.info('selected %i (of %i) features from %s'
            %(vlay.selectedFeatureCount(),vlay.dataProvider().featureCount(), vlay.name()))
        
        res_d = processing.run('native:saveselectedfeatures', 
                                  {'INPUT' : vlay,'OUTPUT' :ofp},  
                                  feedback=self.feedback)
        
        log.info('finished w/ %s'%res_d)
        
        #=======================================================================
        # get tabular data
        #=======================================================================
        if write_csv:
            try:
                #pull the selected data
                df = vlay_get_fdf(vlay, selected=True)
                
                df.to_csv(os.path.join(self.out_dir, '%s_sel.csv'%vlay.name()))
            except Exception as e:
                log.error('failed to extract tabular w/ \n    %s'%e)
        
        return res_d['OUTPUT']


if __name__ == '__main__':
    
    #layer to select features from
    main_fp = r'C:\Users\cefect\Downloads\BritishColumbia.geojson'
     
    #layer to intersect
    top_fp = r'C:\Users\cefect\Downloads\aoi_4326.gpkg'
 
    with Sliceor(tag='LM') as wrkr:
        ofp = wrkr.run_slice(main_fp, top_fp)
    #===========================================================================
    # wrap
    #===========================================================================
    force_open_dir(wrkr.out_dir)
    
    print('finished in %s'%(datetime.datetime.now() - start))

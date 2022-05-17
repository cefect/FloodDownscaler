'''
Created on May 9, 2022

@author: cefect

flood specific Q scripts
'''
import os, datetime
from qgis.core import QgsRasterLayer, QgsMapLayerStore
from hp.Q import Qproj, Error, RasterCalc


class HQproj(Qproj):
    def __init__(self, 
                 dem_fp=None,
                 base_resolution=None,
                 **kwargs):
 
        super().__init__(  **kwargs) 
        
        if not dem_fp is None:
            self.load_dem(fp=dem_fp, base_resolution=base_resolution)
            
    def load_dem(self,
                 fp=None,
                 logger=None,
                 base_resolution=None,
                 ):
        if logger is None: logger=self.logger
        log=logger.getChild('load_dem')
        assert os.path.exists(fp), fp
        
        rlay = self.rlay_load(fp, mstore=self.mstore )
        self.assert_layer(rlay)
        
        #=======================================================================
        # reproject
        #=======================================================================
        if not base_resolution is None:
            if not self.rlay_get_resolution(rlay)==float(base_resolution):
                raise IOError('reproject')
            
        self.dem_rlay = rlay
        
        log.info('loaded DEM as \'%s\' w/ \n    %s'%(
            rlay.name(), self.rlay_get_props(rlay)))
        
    def wse_remove_gw(self, #change all negative depths to null
                      wse_rlay,
                      dem_rlay=None,
 
                      logger=None,
                      out_dir=None,temp_dir=None,
                      ):
        if logger is None: logger=self.logger
        log=logger.getChild('wse_remove_gw')
        if dem_rlay is None: dem_rlay=self.dem_rlay
        if out_dir is None: out_dir=self.out_dir
        if temp_dir is None: temp_dir=self.temp_dir
        mstore=QgsMapLayerStore()
        #=======================================================================
        # precheck
        #=======================================================================
        self.assert_layer(wse_rlay)
        reso = int(self.rlay_get_resolution(dem_rlay))
        #=======================================================================
        # reproject to match
        #=======================================================================
        log.info('warpreproject \'%s\' w/ resolution=%i  '%(wse_rlay.name(), reso))
        fp = self.warpreproject(wse_rlay, compression='none', logger=log,resolution=reso,
                                    resampling='Nearest neighbour', 
                            output=os.path.join(temp_dir, 'preCalc_%s.tif'%wse_rlay.name()))
        
 
        wse1_rlay = self.get_layer(fp, mstore=mstore)
        #=======================================================================
        # build differences
        #=======================================================================
        with RasterCalc(wse1_rlay, name='diff', session=self, logger=log,out_dir=self.temp_dir,) as wrkr:
 
            entries_d = {k:wrkr._rCalcEntry(v) for k,v in {'top':wse1_rlay, 'bottom':dem_rlay}.items()}
            formula = '%s - %s'%(entries_d['top'].ref, entries_d['bottom'].ref)
 
            log.info('executing %s'%formula)
            fp = wrkr.rcalc(formula, layname='%s_diff'%wse_rlay.name())
            
        #=======================================================================
        # build mask
        #=======================================================================
        """building from scratch to be more precise
        dep_rlay = self.get_layer(fp, mstore=mstore)
        self.mask_build(dep_rlay, logger=log, thresh=0.0,
                        logger=log, layname='%s_mask'%wse_rlay.name())"""
        
        with RasterCalc(fp, name='mask', session=self, logger=log,out_dir=self.temp_dir,) as wrkr:
 
            rc = wrkr._rCalcEntry(wrkr.ref_lay)
            
            """deal with some rounding issues?"""
            formula = '(\"{0}\">=0.001)'.format(rc.ref)
 
            log.info('executing %s'%formula)
            fp = wrkr.rcalc(formula, layname='%s_mask'%wse_rlay.name())
            
        #=======================================================================
        # apply mask
        #=======================================================================
        mask_rlay = self.get_layer(fp, mstore=mstore)

        
        ofp = self.mask_apply(wse1_rlay, mask_rlay, layname='%s_noGW'%wse_rlay.name(), 
                              logger=log, out_dir=out_dir)
        
        
        log.info('finished to \n    %s'%ofp)
        return self.get_layer(ofp, mstore=self.mstore)
                        
                        
            

    

 

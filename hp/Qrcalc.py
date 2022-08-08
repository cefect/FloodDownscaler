'''
Created on Aug. 8, 2022

@author: cefect
'''
import time, sys, os, datetime 
from pprint import PrettyPrinter
from qgis.core import QgsMapLayerStore, QgsRasterLayer, QgsCoordinateTransformContext
import numpy as np
import pandas as pd

from qgis.analysis import QgsNativeAlgorithms, QgsRasterCalculatorEntry, QgsRasterCalculator

class RasterCalc(object):
    
    result= None
    layers_d = dict()
    def __init__(self,
                 ref_lay,
                 logger=None,
                 name='rcalc', 
                 session=None,
                 #compression='none',
 
                 out_dir=None,
                 tmp_dir=None,
                 mstore=None,
                 
                 layname=None,
                 ofp=None,
                 ):
        
        #=======================================================================
        # precheck
        #=======================================================================

                
 
        #=======================================================================
        # attach
        #=======================================================================
        
        self.logger = logger.getChild(name)
        self.name=name
        if mstore is None:
            mstore = QgsMapLayerStore()
        self.mstore = mstore
        self.start = datetime.datetime.now()
        #self.compression=compression
        
        #from session
        self.session=session
        self.qproj=session.qproj
        self.feedback=session.feedback
        self.overwrite=self.session.overwrite
        
        #=======================================================================
        # defaults
        #=======================================================================
        self.rasterEntries = list() #list of QgsRasterCalculatorEntry
        #out_dir
        if out_dir is None:
            out_dir = os.environ['TEMP']
            
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir=out_dir
            
        #temp dir
        if tmp_dir is None: tmp_dir=out_dir
        self.tmp_dir=tmp_dir
        
        
        #reference layer
        if isinstance(ref_lay, str):
            ref_lay = self.load(ref_lay)
            
        
        assert isinstance(ref_lay, QgsRasterLayer)
        self.ref_lay=ref_lay
        
        #outputs
        if layname is None: 
            layname='%s_%s'%(ref_lay.name(), self.name)
        assert not '.tif' in layname
        self.layname=layname
        #=======================================================================
        # output file
        #=======================================================================
        if ofp is None:
            ofp = os.path.join(out_dir,layname+'.tif' )
        self.ofp=ofp
        
        self.logger.debug('on ref_lay: %s'%self.ref_lay.name())
    
    def rcalc(self, #simple raster calculations with a single raster
               
               #calc control
               formula, #string formatted formula
               rasterEntries=None, #list of QgsRasterCalculatorEntry
               
               #output control
               ofp=None,
               layname=None,
               #compression=None, #optional compression. #usually we are deleting calc results
               
               #general control
               #allow_empty=True, 
               report=False,
               logger=None,
               ):
        """
        see __rCalcEntry
        
        phantom crashing
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('rcalc1')
 
        #if compression is None: compression=self.compression
        if rasterEntries is None: rasterEntries=self.rasterEntries
        
 
        ref_lay=self.ref_lay
        
        #add the reference to the entries 
        """for simple runs? consider doing this during __init__"""
        if len(rasterEntries)>0:
            self._rCalcEntry(ref_lay)
        
        if layname is None: 
            layname=self.layname
        #=======================================================================
        # output file
        #=======================================================================
        if ofp is None:
            ofp = self.ofp
            
        if os.path.exists(ofp):
            log.debug('file expsts: %s'%ofp) 
            assert self.overwrite
            
            try:
                os.remove(ofp)
            except Exception as e:
                raise IOError('failed to clear existing file.. unable to write \n    %s \n    %s'%(
                    ofp, e))
                
        assert ofp.endswith('.tif')
        
 
        #=======================================================================
        # assemble parameters
        #=======================================================================

        d = dict(
                formula=formula,
                ofp=ofp,
                outputExtent  = ref_lay.extent(),
                outputFormat = 'GTiff',
                crs = self.qproj.crs(),
                nOutputColumns = ref_lay.width(),
                nOutputRows = ref_lay.height(),
                crsTrnsf = QgsCoordinateTransformContext(),
                rasterEntries = rasterEntries,
            )
        #=======================================================================
        # execute
        #=======================================================================
        msg = PrettyPrinter(indent=4).pformat(d)
        #msg = '\n'.join(['%s:    %s'%(k,v) for k,v in d.items()])
        log.debug('QgsRasterCalculator w/ \n%s'%msg)
        
        rcalc = QgsRasterCalculator(d['formula'], d['ofp'],
                                     d['outputFormat'], d['outputExtent'], d['crs'],
                                     d['nOutputColumns'], d['nOutputRows'], d['rasterEntries'], d['crsTrnsf'])
 
 
        
        try:
            result = rcalc.processCalculation(feedback=self.feedback)
        except Exception as e:
            raise IOError('failed to processCalculation w/ \n    %s'%e)
        
        #=======================================================================
        # check    
        #=======================================================================
        if not result == 0:
            raise IOError('formula=%s failed w/ \n    %s'%(formula, rcalc.lastError()))
 
        log.debug('saved result to: \n    %s'%ofp)
 
        #=======================================================================
        # wrap
        #=======================================================================
 
        #check and report
        if report:
            stats_d = self.session.rasterlayerstatistics(ofp)
            log.debug('finished w/ \n    %s'%stats_d)
        self.result = ofp
        return ofp
    
    
    def _rCalcEntry(self, #helper for raster calculations 
                         rlay_obj, bandNumber=1,
 
                         ):
        #=======================================================================
        # load the object
        #=======================================================================
        
        if isinstance(rlay_obj, str):
            rlay = self.load(rlay_obj)
 
        else:
            rlay = rlay_obj
 
        #=======================================================================
        # check
        #=======================================================================
        assert isinstance(rlay, QgsRasterLayer)
        assert rlay.crs()==self.qproj.crs(), 'bad crs on \'%s\' (%s)'%(rlay.name(),rlay.crs().authid())
        
        #=======================================================================
        # build the entry
        #=======================================================================
        rcentry = QgsRasterCalculatorEntry()
        rcentry.raster =rlay #not accesesible for some reason
        rcentry.ref = '%s@%i'%(rlay.name(), bandNumber)
        rcentry.bandNumber=bandNumber
        
        self.rasterEntries.append(rcentry)
        return rcentry

    def load(self, fp, 
 
                  logger=None):
        
        if logger is None: logger = self.logger
        log = logger.getChild('load')
        
        assert os.path.exists(fp), 'requested file does not exist: %s'%fp
        assert QgsRasterLayer.isValidRasterFileName(fp), 'requested file is not a valid raster file type: %s'%fp
        
        basefn = os.path.splitext(os.path.split(fp)[1])[0]
        

        #Import a Raster Layer
        log.debug('QgsRasterLayer(%s, %s)'%(fp, basefn))
        rlayer = QgsRasterLayer(fp, basefn)
 
 
        
        #===========================================================================
        # check
        #===========================================================================
        assert isinstance(rlayer, QgsRasterLayer), 'failed to get a QgsRasterLayer'
        assert rlayer.isValid(), "Layer failed to load!"
        
        
        if not rlayer.crs() == self.qproj.crs():
            log.warning('loaded layer \'%s\' crs mismatch!'%rlayer.name())

        #log.debug('loaded \'%s\' from \n    %s'%(rlayer.name(), fp))
        
        stats_d = self.session.rasterlayerstatistics(rlayer)
        
        assert not pd.isnull(stats_d['MEAN']), 'got a bad layer from \n    %s'%fp
        #=======================================================================
        # wrap
        #=======================================================================
        self.mstore.addMapLayer(rlayer)
        self.layers_d[rlayer.name() ] =rlayer #holding the layer?
        
        return rlayer
    
    #===========================================================================
    # special formulas------
    #===========================================================================
    
    def formula_mbuild(self,
                    ref=None,
                                        #mask parameters
                   zero_shift=False, #necessary for preserving zero values
                   thresh=None, #optional threshold value with which to build raster
                   thresh_type='lower', #specify whether threshold is a lower or an upper bound
                   rval=None, #make a mask from a specific value
                   ):
        #=======================================================================
        # defaults
        #=======================================================================
        if ref is None: ref = self.ref()
        
        assert isinstance(ref, str), ref
        #=======================================================================
        # build formula--
        #=======================================================================
        if rval is None:
            #=======================================================================
            # #from teh data as is
            #=======================================================================
            if thresh is None:

                if zero_shift:
                    """
                    without this.. zero values will show up as null on the mask
                    """
                    f1 = '(abs(\"{0}\")+999)'.format(ref)
                    formula = f1 + '/' + f1
                else: #take any real values
                
                    formula = '\"{0}\"/\"{0}\"'.format(ref)
            #=======================================================================
            # #apply a threshold to the data
            #=======================================================================

            else:
                """
                WARNING: strange treatment of values right around some specified threshold (e.g., rlay<1.01). 
                I think the data has more precision than what is shown (even with using the raster rounder)
                """
                
                if thresh_type=='lower': # val=1 where greater than the lower bound
                    thresh_i = thresh-.001
                    f1 = '(\"{0}\">={1:.3f})'.format(ref, thresh_i)
                    
                elif thresh_type=='upper':
                    thresh_i = thresh+.001
                    f1 = '(\"{0}\"<={1:.3f})'.format(ref, thresh_i)
                elif thresh_type=='lower_neq':
                    f1 = '(\"{0}\">{1:.3f})'.format(ref, thresh)
                elif thresh_type=='upper_neq':
                    f1 = '(\"{0}\"<{1:.3f})'.format(ref, thresh)
                else:
                    raise IOError('bad thresh_type: %s'%thresh_type)
                formula = f1 + '/' + f1
        
        #=======================================================================
        # fixed values
        #=======================================================================
        else:
            """note... this is sensitivite to precision"""
            f1 = '(\"{0}\"={1:.1f})'.format(ref, rval)
            formula = f1 + '/' + f1
            
        return formula
    
    def formula_mapply(self,
                       mask_obj,
                    ref=None,
                    ):
        """apply a mask to the reference
        
        Parameters
        -----------
        mask_obj: str, 
            
        
        mask=1: result = rlay
        mask=0: result = nan
        mask=nan: result=nan
        rlay=nan: result=nan
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if ref is None: ref=self.ref()
        mask = self.get_ref(mask_obj)
        #=======================================================================
        # retrieve mask reference
        #=======================================================================

        return '(\"{raw}\"/\"{mask}\")'.format(raw=ref, mask=mask)
    
 
        
    
    def get_ref_d(self,layers_d):
        entries_d = {k:self._rCalcEntry(v) for k,v in layers_d.items()}            
        return {k:v.ref for k,v in entries_d.items()}
    
    def get_ref(self, obj):
        """intelligent lazy retrival of a QgsRasterCalculatorEntry reference"""
        ref=None
        if isinstance(obj, QgsRasterLayer): 
            ref = self._rCalcEntry(obj).ref
            
        elif isinstance(obj, str):
            if os.path.exists(obj):
                rlay = self.load(obj)
                ref = self._rCalcEntry(rlay).ref
            else:
                ref = obj
            
        else:
            raise TypeError(obj)
            
        
        
        assert isinstance(ref, str)
        
        return ref
 
        
    
    def ref(self):
        return self._rCalcEntry(self.ref_lay).ref
    
    def __enter__(self,*args,**kwargs):
        return self
 
    def __exit__(self, #destructor
                 *args,**kwargs):
         
        #clear your map store
        self.mstore.removeAllMapLayers()
        #print('clearing mstore')
        self.logger.info('finished in %.2f secs w/ %s'%((datetime.datetime.now() - self.start).total_seconds(), self.result))
        #super().__exit__(*args,**kwargs) #initilzie teh baseclass
        

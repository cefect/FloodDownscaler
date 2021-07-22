'''
Created on Jul. 15, 2021

@author: cefect

custom wrappers for whitebox command line
    would be nicer to use the project python libraries
        but these require tkinter
    could be nicer to use the QGIS processing algos
        but I can't get these to setup
'''


import subprocess, os, logging


from hp.dirz import get_temp_dir
from hp.gdal import get_nodata_val

mod_logger = logging.getLogger(__name__)


#===============================================================================
# classes------
#===============================================================================


class Whitebox(object):
    
    exe_fp = r'C:\LS\06_SOFT\whitebox\20210520\WBT\whitebox_tools.exe'
    
    def __init__(self,
                 out_dir=None,
                 logger=mod_logger,
                 overwrite=True,
                 ):
        
        if out_dir is None: out_dir = get_temp_dir()
        self.out_dir=out_dir
        self.logger=logger.getChild('wbt')
        self.overwrite =overwrite

    def breachDepressionsLeastCost(self,
                                   dem_fp, #file path to fill
                                   dist=100, #pixe distance to fill
                                   out_fp = None, #outpath
                                   logger=None,
        
                                   ):
        #=======================================================================
        # defaults
        #=======================================================================
        tool_nm = 'BreachDepressionsLeastCost'
        if logger is None: logger=self.logger
        log=logger.getChild(tool_nm)
        
        if out_fp is None: 
            out_fp = os.path.join(self.out_dir, os.path.splitext(os.path.basename(dem_fp))[0]+'_hyd.tif')

        #=======================================================================
        # configure        
        #=======================================================================
        args = [self.exe_fp,
                '--run={}'.format(tool_nm),
                '--dem={}'.format(dem_fp),
                '--dist=%i'%dist,
                '--min_dist=\'True\'',
                '--fill=\'True\'',
                '--output={}'.format(out_fp),
                '-v'
                ]
        
        log.info('executing \'%s\' on \'%s\''%(tool_nm, os.path.basename(dem_fp)))
        log.debug(args)
        #subprocess.Popen(args)
        #=======================================================================
        # execute
        #=======================================================================
        result = subprocess.run(args, 
                                capture_output=True,text=True,
                                #stderr=sys.STDOUT, stdout=PIPE,
                                ) #spawn process in explorer
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.debug(result.stdout)
        log.info('finished w/ %s'%out_fp)
        
        
        return out_fp
    
    def elevationAboveStream(self,
                             dem_fp,
                             streams_fp,
                             out_fp=None,
                                  logger=None):

        #=======================================================================
        # defaults
        #=======================================================================
        tool_nm = 'ElevationAboveStream'
        if logger is None: logger=self.logger
        log=logger.getChild(tool_nm)
        
        if out_fp is None: 
            out_fp = os.path.join(self.out_dir, os.path.splitext(os.path.basename(dem_fp))[0]+'_HAND.tif')
        
        assert out_fp.endswith('.tif')
 
        #=======================================================================
        # setup
        #=======================================================================
        args = [self.exe_fp,'-v','--run={}'.format(tool_nm),'--output={}'.format(out_fp),
                '--dem={}'.format(dem_fp),
                '--streams={}'.format(streams_fp),
                ]
        
        #=======================================================================
        # execute
        #=======================================================================
        log.info('executing \'%s\' on \'%s\''%(tool_nm, os.path.basename(dem_fp)))
        self.__run__(args) #execute
        
        return out_fp
    
    def fillMissingData(self,
                        rlay_fp,
                        dist=11, #pixel length to infil
                        logger=None, out_fp=None,
                        ):

        #=======================================================================
        # defaults
        #=======================================================================
        tool_nm = 'FillMissingData'
        if logger is None: logger=self.logger
        log=logger.getChild(tool_nm)
        
        if out_fp is None: 
            out_fp = os.path.join(self.out_dir, os.path.splitext(os.path.basename(rlay_fp))[0]+'_fild.tif')
            

        
        #=======================================================================
        # checks
        #=======================================================================
        if os.path.exists(out_fp):
            assert self.overwrite
            os.remove(out_fp)
        assert out_fp.endswith('.tif')
        
        nan_val = get_nodata_val(rlay_fp)
        assert nan_val==-9999,'got unsupported nodata val'
 
        #=======================================================================
        # setup
        #=======================================================================
        args = [self.exe_fp,'-v','--run={}'.format(tool_nm),'--output={}'.format(out_fp),
                '--input={}'.format(rlay_fp),
                '--filter=%i'%dist,
                '--weight=2.0',
                '--no_edges=\'True\'',
                ]
        
        #=======================================================================
        # execute
        #=======================================================================
        log.info('executing \'%s\' on \'%s\''%(tool_nm, os.path.basename(rlay_fp)))
        self.__run__(args) #execute
        
        return out_fp
    
    def IdwInterpolation(self,
                        vlay_pts_fp, fieldn, 
                        weight=2, #IDW weight value
                        cell_size=10, 
                        logger=None, out_fp=None,
                        ):

        #=======================================================================
        # defaults
        #=======================================================================
        tool_nm = 'IdwInterpolation '
        if logger is None: logger=self.logger
        log=logger.getChild(tool_nm)
        
        if out_fp is None: 
            out_fp = os.path.join(self.out_dir, os.path.splitext(os.path.basename(vlay_pts_fp))[0]+'_idw.tif')
        
        assert out_fp.endswith('.tif')
 
        #=======================================================================
        # setup
        #=======================================================================
        args = [self.exe_fp,'-v','--run={}'.format(tool_nm),'--output={}'.format(out_fp),
                '--input={}'.format(vlay_pts_fp),
                '--field=%s'%fieldn,
                '--weight=%.2f'%weight,
                '--cell_size=%.2f'%cell_size,
                ]
        
        #=======================================================================
        # execute
        #=======================================================================
        log.info('executing \'%s\' on \'%s\''%(tool_nm, os.path.basename(vlay_pts_fp)))
        self.__run__(args) #execute
        
        return out_fp
        
    def __run__(self, args):
        self.logger.debug(args)
        result = subprocess.run(args,capture_output=True,text=True,) 
        self.logger.debug(result.stdout)
        return result




if __name__ == '__main__':
    dem_fp = r'C:\LS\03_TOOLS\_jobs\202103_InsCrve\outs\HAND\HRDEM_cilp2.tif'
    result = Whitebox().breachDepressionsLeastCost(dem_fp)
    
    
    print('finished')




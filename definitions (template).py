'''
Created on Mar. 25, 2022

@author: cefect

I guess this is overwritten by local projects?
'''
import os
 



src_dir = os.path.dirname(os.path.abspath(__file__))
src_name = os.path.basename(src_dir)

#location of logging configuration file
logcfg_file=os.path.join(src_dir, r'coms\logger.conf')

#default working directory
wrk_dir = r'L:\10_IO\coms'

#spatial (mostly for testing)
epsg = 3857
bounds = (0, 0, 100, 100)

 

 

 


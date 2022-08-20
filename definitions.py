'''
Created on Mar. 25, 2022

@author: cefect

I guess this is overwritten by local projects?
'''
import os
 

logcfg_file=r'C:\LS\09_REPOS\01_COMMON\coms\logger.conf'

src_dir = os.path.dirname(os.path.abspath(__file__))

#default working directory
wrk_dir = r'C:\LS\10_IO\coms'

 

src_name = os.path.basename(src_dir)


whitebox_exe_d = {
        'v1.4.0':r'C:\LS\06_SOFT\whitebox\v1.4.0\whitebox_tools.exe',
        'v2.0.0':r'C:\LS\06_SOFT\whitebox\v2.0.0\whitebox_tools.exe',
        }

 
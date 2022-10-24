'''
Created on Aug. 21, 2022

@author: cefect
'''
import os, pstats, cProfile

from definitions import wrk_dir
from hp.dirz import get_valid_filename

def prof_simp(exe_str):
    #build stats file
    out_dir = os.path.join(wrk_dir, 'profile')
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    stats_fp = os.path.join(out_dir, get_valid_filename('cProfile_stats_%s'%exe_str))
    
    #execute profile
    cProfile.run(exe_str, filename=stats_fp)    
    print('finished profile to \n    %s'%stats_fp)
    
    #print profile stats
    p = pstats.Stats(stats_fp)
    p.strip_dirs().sort_stats('cumulative').print_stats(30)
    
    
def stats_view(
        stats_fp= r'C:\LS\10_IO\2207_fpolish\profile\stats_test',
        ):
    
    p = pstats.Stats(stats_fp)
    
    p.strip_dirs().sort_stats('cumulative').print_stats(30)
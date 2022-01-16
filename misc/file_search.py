'''
Created on Jul. 26, 2021

@author: cefect

Finds a list of filenames within a directory
    then copies the found files to the output
'''
import os, shutil
from hp.exceptions import Error
ext='.zip'

search_dir = r'C:\LS\05_DATA\Canada\GOC\NRCan\FloodsInCanada'
out_dir = r'C:\LS\02_WORK\02_Mscripts\InsuranceCurves\06_DATA\FloodsInCanada\20210924'
fn_srch_l = [
    'FloodExtentPolygon_AB_LowerAthabasca_20200428_181633',
    'FloodExtentPolygon_QC_LowerOttawa_20190426_225139',
    'FloodExtentPolygon_NB_LowerSaintJohn_20180504_103055',
    'FloodExtentPolygon_QC_CentralOttawa_20190516_231035',
    'FloodExtentPolygon_NB_LowerSaintJohn_20190423_103028',
    'FloodExtentPolygon_QC_GrandMontreal_20190425_044205',
    'FloodExtentPolygon_QC_GrandMontreal_20170509_111737',
    'FloodExtentPolygon_NB_LowerSaintJohn_20180503_103539',
    'FloodExtentPolygon_NB_LowerSaintJohn_20180507_101859',
    'FloodExtentPolygon_QC_CentralOttawa_20190503_113004',
    'FloodExtentPolygon_QC_Kingston_20190429_230652',
    'FloodExtentPolygon_QC_LacDeuxMontagnes_20170504_105237',
    'FloodExtentPolygon_QC_LacDeuxMontagnes_20170506_225432',
    'FloodExtentPolygon_QC_LowerOttawa_20190430_111737',
    'FloodExtentPolygon_QC_RiviereGatineau_20170509_230723' 
    ]


assert os.path.exists(out_dir)
assert os.path.exists(search_dir)


    

def get_fn_match( #search for a filename in a list of filenames
        search_fn, fps_all,
        ):
    
    for fp in fps_all:
        bdir, fn = os.path.split(fp)
        if search_fn in fn:
            return fp
    
    raise Error('failed to find match in %i for \n    %s'%(len(fps_all), search_fn))
    


def copyit(fp):
    assert os.path.exists(fp)
    ofp = os.path.join(out_dir, os.path.basename(fp))
    
    print("copying \n   from: %s \n    to: %s"%(
        fp, ofp))
    
    assert not os.path.exists(ofp)
    
    shutil.copy2(fp,ofp)
    
    return ofp

#===============================================================================
# collect all filenames
#===============================================================================

fps_all = set()
for dirpath, _, fns in os.walk(search_dir):
    fps_all.update([os.path.join(dirpath, e) for e in fns if e.endswith(ext)])
                
#fn_all = [e for sub_l in [v[2] for v in  os.walk(search_dir)] for e in sub_l]

#===============================================================================
# find matches
#===============================================================================
print('searching for matches on %i files in %i found'%(len(fn_srch_l), len(fps_all)))

res_d = dict()
for search_fn in fn_srch_l:
    match_fp = get_fn_match(search_fn, fps_all)
    
    ofp = copyit(match_fp)
    
    res_d[search_fn] = {'raw_fp':match_fp, 'new_fp':ofp}
    



print('finished on %i'%len(res_d))

if not len(res_d)== fn_srch_l:
    raise Error('failed to match all')



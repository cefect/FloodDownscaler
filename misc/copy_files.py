'''
Created on Jul. 26, 2021

@author: cefect

copying files in a dictionary to some other directory
'''
import os, shutil

out_dir = r'C:\LS\03_TOOLS\_jobs\202103_InsCrve\outs\DR\Fred12_0724'
fp_d = {
    
            'dem_fp':r'C:\LS\03_TOOLS\_jobs\202103_InsCrve\outs\depWf\fred01_0723\hrdem_Fred01_0722_05_fild.tif',
            'fic_fp':r'C:\LS\03_TOOLS\_jobs\202103_InsCrve\outs\depWf\fred01_0723\FiC_Fred01_1x_20180502-20180504_072223.gpkg',
            'nhn_raw_fp':r'C:\LS\03_TOOLS\_jobs\202103_InsCrve\outs\depWf\fred01_0723\NHN_HD_WATERBODY_Fred01_0723_raw.gpkg',
            'nhn_fp':r'C:\LS\03_TOOLS\_jobs\202103_InsCrve\outs\depWf\fred01_0723\NHN_HD_WATERBODY_Fred01_0723_clean.gpkg',
            'hand_fp':r'C:\LS\03_TOOLS\_jobs\202103_InsCrve\outs\depWf\fred01_0723\Fred01_DR_0723_HAND.tif',
            'ndb_fp':r'C:\LS\03_TOOLS\_jobs\202103_InsCrve\outs\depWf\fred01_0723\Fred01_DR_0723_ndb.gpkg',
    }


assert os.path.exists(out_dir)


def copyit(fp):
    assert os.path.exists(fp)
    ofp = os.path.join(out_dir, os.path.basename(fp))
    
    print("copying \'%s\'  \n    from: %s \n    to: %s"%(
        tag, fp, ofp))
    
    assert not os.path.exists(ofp)
    
    shutil.copy2(fp,ofp)
    
    return ofp

for tag, fp in fp_d.items():
    try:
        copyit(fp)
    except Exception as e:
        print('failed to copy %s w/ \n    %s'%(tag, e))
    
    



print('finished on %i'%len(fp_d))
'''
Created on Sep. 8, 2022

@author: cefectv

view pickle files
'''

def open_pick(fp):
    import pandas as pd
    from hp.pd import view
    df = pd.read_pickle(fp)
    
    print('loaded %s\n    cols:%s\n    index:%s'%(
        str(df.shape), df.columns, df.index))
 
    view(df)
    
    
if __name__ == "__main__":
    fp = r'C:\Users\cefect\AppData\Local\Temp\pytest-of-cefect\pytest-2863\test_02_dsc_C___LS__09_REPOS__0\cMasks\SJ_test02_direct_0910_cMasks.pkl'
    
    open_pick(fp)
'''
Created on Sep. 8, 2022

@author: cefectv

view pickle files
'''

def open_pick(fp):
    import pandas as pd
    from hp.pd import view
    df = pd.read_pickle(fp)
 
    view(df)
    
    
if __name__ == "__main__":
    fp = r'C:\Users\cefect\AppData\Local\Temp\pytest-of-cefect\pytest-2807\test_runHaz_proj_d0_filter_dsc0\outs\agg2\tRn\tCn\filter\20220909\errStats\tCn_tRn_filter_0909_errStats.pkl'
    
    open_pick(fp)
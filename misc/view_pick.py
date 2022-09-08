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
    fp = r'C:\LS\10_OUT\2112_Agg\outs\SJ\r4_filter\20220908\errStats\SJ_r4_filter_0908_errStats.pkl'
    
    open_pick(fp)
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
    fp = r'C:\LS\10_OUT\2112_Agg\outs\agg2\r7\SJ\direct\20220910\arsc\SJ_r7_direct_0910_arsc.pkl'
    
    open_pick(fp)
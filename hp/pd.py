'''
Created on Mar 11, 2019

@author: cef

pandas functions 

py3.7

pd.__version__

'''
import logging
import numpy as np
import pandas as pd
import pandas.testing as pdt
import warnings

#===============================================================================
# pandas global options
#===============================================================================
pd.options.mode.chained_assignment = None   #setting with copy warning handling

#truncate thresholds
#===============================================================================
# pd.set_option("display.max_rows", 20)
# 
# pd.set_option("display.max_columns", 10)
# pd.set_option("display.max_colwidth", 12)
# 
# #truncated views
# pd.set_option("display.min_rows", 15)
# pd.set_option("display.min_rows", 15)
# pd.set_option('display.width', 150)
#===============================================================================

 
#===============================================================================
# custom imports
#===============================================================================

from hp.exceptions import Error
#import hp.np
#from hp.np import left_in_right as linr



#mod_logger = logging.getLogger(__name__) #creates a child logger of the root

bool_strs = {'False':False,
             'false':False,
             'FALSE':False,
             0:False,
             'True':True,
             'TRUE':True,
             'true':True,
             1:True,
             False:False,
             True:True}


#===============================================================================
#VIEWS ---------------------------------------------------------
#===============================================================================



def view_web_df(df):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    import webbrowser
    #import pandas as pd
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        #type(f)
        df.to_html(buf=f)
        
    webbrowser.open(f.name)
    
def view(df):
    view_web_df(df)
    
  
    
#===============================================================================
#BOOLEANS ------------------------------------------------------------------
#===============================================================================
def get_bx_multiVal(df, #get boolean based on multi-column matching (single values)
        val_d, #{key:value} for making selection
        logicFunc = np.logical_and, #function for combining iterations
        baseBoolean=True, #where to start from
        matchOn='columns', #here to perform the value matching
        log=None,
        ):
    
    #===========================================================================
    # default
    #===========================================================================
    
    
    
    #===========================================================================
    # construct the serach frame
    #===========================================================================

    if matchOn=='columns':
        pass

    elif matchOn=='index':
        df_raw = df.copy()
        #checks
        mdex = df.index
        assert isinstance(mdex, pd.MultiIndex)
        
        miss_l = set(val_d.keys()).difference(mdex.names)
        assert len(miss_l)==0, miss_l
        
        df = mdex.to_frame().reset_index(drop=True)
        
        
    else:
        raise Error(matchOn)
    
    bx = pd.Series(baseBoolean, index=df.index) #start with nothing
    meta_d= {'base':{'bx':bx.sum()}, 'logicFunc':logicFunc.__name__}
    #===========================================================================
    # execute serach
    #===========================================================================
    for coln, val in val_d.items():
        if isinstance(val, list):
            new_bx = df[coln].isin(val)
            
        elif isinstance(val, tuple):
            raise Error('not implemented')
        else:
            new_bx = df[coln]==val
            
        bx = logicFunc(bx,new_bx)
        
        meta_d[coln] = {'val':val, 'new_bx':new_bx.sum(), 'bx':bx.sum()}
        
    #===========================================================================
    # wrap
    #===========================================================================
    if not log is None:
        log.info('on %s w/ %i matching vals got %i/%i for %s'%(
            str(df.shape), len(val_d),  bx.sum(), len(bx), val_d))
        
        mdf = pd.DataFrame.from_dict(meta_d).T
        log.debug(mdf)
        
    #reset the multindex
    if matchOn=='index':
        bx.index = df_raw.index
            
    return bx
        
 
#===============================================================================
# MULTINDEX-----------
#===============================================================================
def append_levels(mdex, keys_d):
    """append dummy levels"""
    
    for k,v in keys_d.items():
        assert not k is None, k
    
    df = mdex.to_frame().reset_index(drop=True)
    for name, level in keys_d.items():
        df[name] = level
        
    df = df.loc[:, list(keys_d.keys())+ list(mdex.names)] #rerder
    
    return pd.MultiIndex.from_frame(df)
#===============================================================================
# MISC --------------------------------------------------------------------
#===============================================================================
 
    



#===============================================================================
# assertions
#===============================================================================
def assert_index_equal(*args, msg='', **kwargs):
    if __debug__:
        try:
            pdt.assert_index_equal(*args, **kwargs)
        except Exception as e:
            raise AssertionError('%s\n%s'%(msg, e))
        
        
if __name__ == '__main__':
    pass
    #
 
    
    


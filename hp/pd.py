'''
Created on Mar 11, 2019

@author: cef

pandas functions 

py3.7

pd.__version__

'''
import logging, copy, os, time, re, xlrd, math, gc, inspect
import numpy as np
import pandas as pd
import warnings

#===============================================================================
# pandas global options
#===============================================================================
pd.options.mode.chained_assignment = None   #setting with copy warning handling

#truncate thresholds
pd.set_option("display.max_rows", 20)
pd.set_option("display.max_colwidth", 20)

#truncated views
pd.set_option("display.min_rows", 15)
pd.set_option("display.min_rows", 15)
pd.set_option('display.width', 100)

 
#===============================================================================
# custom imports
#===============================================================================

from hp.exceptions import Error
#import hp.np
#from hp.np import left_in_right as linr



mod_logger = logging.getLogger(__name__) #creates a child logger of the root

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
 
#===============================================================================
# MISC --------------------------------------------------------------------
#===============================================================================
 
    

def data_report( #generate a data report on a frame
        df,
        ofp = None, #Optional filename for writing the report xls to file
        
        
        
        include_df = False, #whether to include the full dataset
        
        #value report selection
        val_rpt=True,
        skip_unique = True, #whether to skip attribute value count p ublishing on unique values
        max_uvals = 500, #maximum number of unique value check 
        
        #value report behavcior
        vc_dropna = False, #whether to drop nas from the value count tabs
        
        logger = mod_logger):
    warnings.warn('2021-12-13', DeprecationWarning)
    #===========================================================================
    # setup
    #===========================================================================
    log = logger.getChild('data_report')
    
    #setup results ocntainer
    res_df = pd.DataFrame(index = df.columns, columns=('empty','dtype', 'isunique','unique_vals', 'nulls', 'reals', 'real_frac', 'mode'))
    
    
    res_d = dict() #empty container for unique values
    
    #===========================================================================
    # loop and calc
    #===========================================================================
    for coln, col_ser in df.iteritems():
        log.debug('collecting data for \'%s\''%coln)
        res_df.loc[coln, 'empty'] = len(col_ser) == col_ser.isna().sum()

        #type
        res_df.loc[coln, 'dtype'] = str(col_ser.dtype.name)
        
        #unique
        res_df.loc[coln, 'isunique'] = str(col_ser.is_unique)
        
        #unique values
        uq_vals_cnt = len(col_ser.unique())
        res_df.loc[coln, 'unique_vals'] = uq_vals_cnt
        
        #nulls
        res_df.loc[coln, 'nulls'] = col_ser.isna().sum()
        res_df.loc[coln, 'reals'] = len(col_ser) - col_ser.isna().sum()
        
        res_df.loc[coln, 'real_frac'] =  float((len(col_ser) - col_ser.isna().sum()))/float(len(col_ser))
        
        #mode
        if len(col_ser.mode()) ==1:
            res_df.loc[coln, 'mode'] = col_ser.mode()[0]

        
        #=======================================================================
        # float reports
        #=======================================================================
        
        if np.issubdtype(col_ser.dtype, np.number):
            res_df.loc[coln, 'min']=col_ser.min()
            res_df.loc[coln, 'max']=col_ser.max()
            res_df.loc[coln, 'mean']=col_ser.mean()
            res_df.loc[coln, 'median']=col_ser.median()
            res_df.loc[coln, 'sum']=col_ser.sum()
            

            
        
        #=======================================================================
        # value reports
        #=======================================================================
        if not val_rpt: continue
        #unique ness check
        if skip_unique and col_ser.is_unique:
            log.warning('skipping val report for \'%s\''%coln)
            continue
        
        #ratio check
        if uq_vals_cnt > max_uvals:

            log.info('skippin val report for \'%s\' unique vals (%i) > max (%i)'%(
                coln, uq_vals_cnt, max_uvals))
            continue

        vc_df = pd.DataFrame(col_ser.value_counts(dropna=vc_dropna))
        
        if len(vc_df)> 0:
            res_d[coln] = vc_df
        else:
            """shouldnt trip if dropna=True?"""
            log.warning('got no value report for \'%s\''%coln)
        
        
    #===========================================================================
    # wrap up
    #===========================================================================
    #create a new dict with this at the front
    res_d1 = {'_smry':res_df} 
    res_d1.update(res_d)
    
    if include_df:
        res_d1['data'] = df

    """
    res_d1.keys()
    """
    #===========================================================================
    # write
    #===========================================================================
    if not ofp is None:
        log.debug('sending report to file:\n    %s'%ofp)
        hp.pd.write_to_xls(ofp, res_d1, logger=log, allow_fail=True)
    
    
    return res_d1

        
        
if __name__ == '__main__':
    pass
    #
 
    
    


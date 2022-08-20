'''
Created on Mar 5, 2019

@author: cef

 
'''



#===============================================================================
# # imports --------------------------------------------------------------------
#===============================================================================
import numpy as np
import warnings
 

def get_all_blocks(a, n=2):
    """generate 2D blocks"""
    for i in range(a.shape[0] // n):
        for j in range(a.shape[1]//n):
            yield a[n*i:n*(i+1), n*j:n*(j+1)]
    
        
def get_flat_blocks(a, n=2, errors='raise'):
    """block the array then build a new array where each row is a flat block
    
    surprised there is no builtin..."""
    
    #check for uniform blocks
    errs = list()
    for i, dim in enumerate(a.shape):
        if not dim%n==0:
            errs.append('axis %i has bad split: %.2f (shape needs to be a multiple of n)'%(i, dim%n))
            
    if len(errs)>0:
        if errors=='raise':
            raise IndexError(errs)
        elif errors=='warn':
            warnings.warn(errs)
    
    res_ar = np.array([sa.flatten() for sa in get_all_blocks(a, n=n)])
    
    #post check
    assert res_ar.shape[1]==n**2
    assert res_ar.shape[0]==int(np.array(res_ar.shape).prod()/(n**2))
    
    return res_ar

def apply_blockwise_ufunc(a, ufuncName, n=2):
    """apply a numpy ufunc to 2d blocks
    
    Parameters
    ----------
    a: np.array
        raw array
    ufunc: str
        name of numpy ufunc
    n: int, default 2
        dimension for square block
        
    """
    #broadcast each square block as a row
    blocked_ar = get_flat_blocks(a, n=n)
    
    #get this ufunc
    ufunc = getattr(np, ufuncName)
    
    #apply the reduction
    bred_ar = ufunc.reduce(blocked_ar.T)
    
    #recast to match raw shape (reduced)    
    new_shape = [int(e) for e in np.fix(np.array(a.shape)/n).tolist()]
    res_ar = bred_ar.reshape(new_shape)
    
    
    assert np.array_equal(np.array(res_ar.shape)*n,np.array(a.shape)) 
    
    
    
    return res_ar

 

def apply_blockwise(a, func,n=2, **kwargs):
    """apply a numpy ufunc to 2d blocks
    
    Parameters
    ----------
    a: np.array
        raw array
    func: numpy method to apply
        must take an array and an axis kwarg
    n: int, default 2
        dimension for square block
        """
    #broadcast each square block as a row
    blocked_ar = get_flat_blocks(a, n=n)
    
 
    #apply the reduction
    bred_ar = func(blocked_ar, axis=1, **kwargs)
    
    #recast to match raw shape (reduced)    
    new_shape = [int(e) for e in np.fix(np.array(a.shape)/n).tolist()]
    res_ar = bred_ar.reshape(new_shape)
    
    
    assert np.array_equal(np.array(res_ar.shape)*n,np.array(a.shape))    
    
    return res_ar
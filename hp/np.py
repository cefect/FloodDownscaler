'''
Created on Mar 5, 2019

@author: cef

 
'''



#===============================================================================
# # imports --------------------------------------------------------------------
#===============================================================================
import numpy as np
import warnings

np.set_printoptions(linewidth=200)
#import skimage.transform
#from scipy.ndimage import uniform_filter, generic_filter, zoom
#===============================================================================
# np.set_printoptions(edgeitems=10,linewidth=180)
# np.set_printoptions(edgeitems=10)
# np.core.arrayprint._line_width = 180
#===============================================================================

#===============================================================================
# def get_all_blocks(a, n=2):
#     """generate 2D blocks"""
#     for i in range(a.shape[0] // n):
#         for j in range(a.shape[1]//n):
#             yield a[n*i:n*(i+1), n*j:n*(j+1)]
#     
#         
# def get_flat_blocks(a, n=2, errors='raise'):
#     """block the array then build a new array where each row is a flat block
#     
#     surprised there is no builtin..."""
#     
#     if __debug__:
#         #check for uniform blocks
#         errs = list()
#         for i, dim in enumerate(a.shape):
#             if not dim%n==0:
#                 errs.append('axis %i has bad split: %.2f (shape needs to be a multiple of n)'%(i, dim%n))
#                 
#         if len(errs)>0:
#             if errors=='raise':
#                 raise IndexError(errs)
#             elif errors=='warn':
#                 warnings.warn(errs)
#     
#     res_ar = np.array([sa.flatten() for sa in get_all_blocks(a, n=n)])
#     
#     #post check
#     assert res_ar.shape[1]==n**2
#     assert res_ar.shape[0]==int(np.array(res_ar.shape).prod()/(n**2))
#     
#     return res_ar
#===============================================================================

#===============================================================================
# def apply_blockwise_ufunc(a, ufuncName, n=2):
#     """apply a numpy ufunc to 2d blocks
#     
#     Parameters
#     ----------
#     a: np.array
#         raw array
#     ufunc: str
#         name of numpy ufunc
#     n: int, default 2
#         dimension for square block
#         
#     """
#     #broadcast each square block as a row
#     blocked_ar = get_flat_blocks(a, n=n)
#     
#     #get this ufunc
#     ufunc = getattr(np, ufuncName)
#     
#     #apply the reduction
#     bred_ar = ufunc.reduce(blocked_ar.T)
#     
#     #recast to match raw shape (reduced)    
#     new_shape = [int(e) for e in np.fix(np.array(a.shape)/n).tolist()]
#     res_ar = bred_ar.reshape(new_shape)
#     
#     
#     assert np.array_equal(np.array(res_ar.shape)*n,np.array(a.shape)) 
#     
# 
#     return res_ar
#===============================================================================

 

def apply_block_reduce(a, func,aggscale=2):
    """apply a reducing function to square blocks (window w/o overlap)
    
    Parameters
    ----------
    a: np.array
        raw array
    func: numpy method to apply
        must take an array and an axis kwarg
    aggscale: int, default 2
        reducer for new shape
        
    Note
    --------
    spent a few hours looking for native methods. 
    
    https://stackoverflow.com/questions/73529481/numpy-based-spatial-reduction/73529581#73529581
        """
    #===========================================================================
    # defaults
    #===========================================================================
    assert isinstance(aggscale, int)
    assert aggscale>1
    
    new_shape = (a.shape[0]//aggscale, a.shape[1]//aggscale)
    
    """doesnt seem to work for 2D windows
    #===========================================================================
    # np.stride_tricks
    #===========================================================================
    new_shape = (a.shape[0]//aggscale, a.shape[1]//aggscale)
    
    a.flatten
        
    nrows = ((a.size-n)//n)+1
    a.strides
    strides = a.strides[0]
        

    
    np.lib.stride_tricks.as_strided(a, shape=new_shape, strides=(aggscale, aggscale))"""
    
    """
    #===========================================================================
    # scipy.ndimage
    #===========================================================================
    #uniform_filter(a, size=aggscale, mode='constant', cval=0.0)
    
    generic_filter(a, func, size=aggscale, mode='constant', cval=0.0)
    
    #build the mask
    afi = np.full((aggscale, aggscale), 0)
    afi[-1,-1]=1    
    mask =   np.tile(np.tile(afi, a.shape[1]//aggscale).T, a.shape[0]//aggscale)
    """
    
    #===========================================================================
    # np.reshape
    #===========================================================================
    #stack windows into axis 1 and 3
    a1 = a.reshape(a.shape[0]//aggscale, aggscale, a.shape[1]//aggscale, aggscale)
    
 
    res_ar2=func(a1, axis=(1,3))
    
    
 
    #===========================================================================
    # using custom block functions
    #===========================================================================
    """quite slow for loops..."""
    #===========================================================================
    # #broadcast each square block as a row
    # blocked_ar = get_flat_blocks(a, n=aggscale)
    # 
    # #apply the reduction
    # bred_ar = func(blocked_ar, axis=1, **kwargs)
    # 
    # #recast to match raw shape (reduced) 
    # res_ar = bred_ar.reshape(new_shape)
    # 
    # 
    # assert np.array_equal(np.array(res_ar.shape)*aggscale,np.array(a.shape))
    # 
    # assert np.array_equal(res_ar2, res_ar)    
    #===========================================================================
    
    assert res_ar2.shape==new_shape
    
    return res_ar2

def downsample(a, n=2):
    """increase shape. scale up an array by replicating parent cells onto children with spatial awareness
    
    Parameters
    ----------
    n: int, default 2
        amount to scale shape by
    
    very confusing.. surprised there is no builtin"""
    
    
    
    assert isinstance(n, int)
    assert n>1
    new_shape = (a.shape[0]*n, a.shape[1]*n)
    
    """runs out of memory
    #===========================================================================
    # np.kron
    #===========================================================================
    
    np.kron(a, np.ones((n,n)))"""
    
    """interploates
    #===========================================================================
    # scipy.ndimage.zoom
    #===========================================================================
    zoom(a, n, """
    
    #===========================================================================
    # scipy.ndimage.zoom
    #===========================================================================
    """preferred method"""
    raise IOError('use ndimage.zoom in place')
    scipy.ndimage.zoom(mar_raw, scale, order=0, mode='reflect',   grid_mode=True)
    
    #===========================================================================
    # skimage.transform.resize
    #===========================================================================
 #==============================================================================
 #    """seems to work.. should be a faster way though w/o polynomial"""
 #    
 #    res_ar2 = skimage.transform.resize(a, new_shape, order=0, mode='constant')
 #    res_ar = res_ar2
 #    
 #    """tiles blocks... doesn't zoom
 #    #===========================================================================
 #    # np.tile
 #    #===========================================================================
 #    #np.tile(np.tile(a, n).T, a.shape[0]//downscale)"""
 #    
 #    """
 #    #===========================================================================
 #    # concat list
 #    #===========================================================================
 #    
 #    l=list()
 #    for i in range(a.shape[0]):
 #        #=======================================================================
 #        # l = list()
 #        # for b in build_blocks(a[i, :].reshape(-1), n=n):
 #        #     l.append(b)
 #        #=======================================================================
 # 
 #        
 #        new_ar = np.concatenate([b for b in build_blocks(a[i, :].reshape(-1), n=n)], axis=1)
 #        #print('i=%i\n%s'%(i, new_ar))
 #        l.append(new_ar)
 #    
 #    res_ar = np.concatenate(l, axis=0) 
 #    assert np.array_equal(res_ar2, res_ar)
 #    """
 #    
 #    assert res_ar.shape==new_shape
 #    
 #    return res_ar
 #==============================================================================


#===============================================================================
# def xxxupsample2(a, n=2):
#     """scale up an array by replicating parent cells onto children with spatial awareness
#     
#     using apply
#     
#     this is slower!"""
#     
#     
#     
#     def row_builder(a1, n=2):
#         row_ar = np.concatenate([b for b in build_blocks(a1, n=n)], axis=1)
#         return row_ar
#  
#     """only useful for 3d it seems
#     np.apply_over_axes(row_builder, a, [0,1])"""
#     
#     #build blocks for each row (results are stacked as in 3D)
#     res3d_ar = np.apply_along_axis(row_builder, 1, a, n=n)
#     
#     #split out each 3D and recombine horizontally
#     res_ar = np.hstack(np.split(res3d_ar, res3d_ar.shape[0], axis=0))[0]
#  
#     #check
#     new_shape = tuple([int(e) for e in np.fix(np.array(a.shape)*n).tolist()])
#     assert res_ar.shape==new_shape
#     return res_ar
#         
#      
# 
# def build_blocks(a, n=2):
#     """generate 2D blocks"""
#     for x in np.nditer(a):
#         yield np.full((n,n), x)
#===============================================================================
        
def dropna(a):
    """mimic pandas behavior"""
    return a[~np.isnan(a)]

'''
Created on Dec. 4, 2022

@author: cefect

basic... not accelerated reduction
'''
import numpy as np

#===============================================================================
# def disag(ar, downscale=2):
#     """disaggregate/downscale the array to the specified scale
#     
#     this replicates basic behavior of 
#         scipy.ndimage.zoom
#         
#     stacks into 4 dimensions to handle 2-D rescaling
#     """
#     #rescaled 2D shape
#     new_shape=tuple([int(e*downscale) for e in ar.shape])
#     
#     #shape of 4D stack
#     shape4=tuple([new_shape[0]//downscale, downscale, new_shape[1]//downscale, downscale])
#     
#     l=list()
#     for e in ar:
#         l.append(np.broadcast_to(e, (2,4)))
#         
#     np.array(l)
#     ar[0]
#     np.broadcast_to(ar, (2,4,2,4))
#     
#     np.broadcast_to(ar[0], (2,4))
#     
#     np.broadcast_to(ar, (4,4,4))
#     
#     np.broadcast_to(ar, (4,4,4))
#     
#     
#     #build the result container
#     res_ar = np.full(new_shape, np.nan)
#     #res_ar = np.arange(new_shape[0]*new_shape[1]).reshape(new_shape)
#     
#     #reshape    
#     res_ar1 = res_ar.reshape(shape4) 
#===============================================================================
    
    
    

def disag(ar, downscale=2):
    """disaggregate/downscale the array to the specified scale
    
    this replicates basic behavior of 
        scipy.ndimage.zoom
        
    stacks into 4 dimensions to handle 2-D rescaling
    
    
    I suspect there is a way to do this with just broadcast commands...
        but I couldn't figure it out
    """
    #rescaled 2D shape
    new_shape=tuple([int(e*downscale) for e in ar.shape])
    
    #shape of 4D stack
    shape4=tuple([new_shape[0]//downscale, downscale, new_shape[1]//downscale, downscale])
 
    
    
    #build the result container
    res_ar = np.full(new_shape, np.nan)
    #res_ar = np.arange(new_shape[0]*new_shape[1]).reshape(new_shape)
    
    #reshape    
    res_ar1 = res_ar.reshape(shape4) 
    
    #build iterator
    it = np.nditer([ar, res_ar1],
            flags = [
                #'external_loop', 
                #'buffered'
                'multi_index'
                ],
            op_flags = [['readonly'],
                        ['writeonly', 
                         #'allocate', #populate None 
                         'no_broadcast', #no aggregation?
                         ]],
            #op_axes=[[0,-1,-1,-1], [0,1,2,3]], #just the first column of ar
            op_axes=[[0,-1,1,-1], [0,1,2,3]],  
            )
    
    #===========================================================================
    # execute iteration
    #===========================================================================
    with it:
        #it.operands[1][...] = -9999
        for x, y in it:
            #y[...] = x*x
            #print(f'x={x} y={y} ({it.multi_index})')
            y[...] = x
        result= it.operands[1]
        
 
    return result.reshape(new_shape) 
    

def disag_nditer(ar, downscale=2):
    """disaggregate/downscale the array to the specified scale
    
    results in an array with scale = ar.shape*downscale
    """
    assert isinstance(downscale, int)
    new_shape=[int(e*downscale) for e in ar.shape]
    res_ar = np.full(new_shape, np.nan)
    
    
    it = np.nditer([ar, res_ar],
            flags = [
                #'external_loop', 
                #'buffered'
                ],
            op_flags = [['readonly'],
                        ['writeonly', 
                         #'allocate', #populate None 
                         #'no_broadcast', #no aggregation?
                         ]],
            #op_axes=[None, new_shape],
            )
                         
    with it:
        #it.operands[1][...] = -9999
        for x, y in it:
            #y[...] = x*x
            y[...] = np.full((downscale,downscale), x)
        return it.operands[1]
    
    for x in np.nditer(ar, 
                       #flags=['external_loop']
                       ):
        print(x, end=',')
    

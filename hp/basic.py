'''
Created on Dec. 29, 2021

@author: cefect
'''


def set_info( #get all the  membership info from two containers
        left, right,
             result='elements'):
    
    el_d =  {
        'diff_left':set(left).difference(right), #those in left no tin right
        'diff_right':set(right).difference(left),
        'union':set(left).union(right),
        'intersection':set(left).intersection(right),
        'symmetric_difference':set(left).symmetric_difference(right),        
        }
    
    if result=='elements':
        return el_d
    elif result=='counts':
        return {k:len(v) for k,v in el_d.items()}
    else:
        raise ValueError('unrecognized results key \'%s\''%result)
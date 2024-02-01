'''
Created on Sep. 17, 2022

@author: cefect
'''

import itertools
from itertools import repeat
import multiprocessing

from rasterstats import zonal_stats

def _zonal_stats_partial(feats, rlay_fp, **kwargs):
    """Wrapper for zonal stats, takes a list of features"""
    return zonal_stats(feats, rlay_fp, **kwargs)



def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def zonal_stats_multi(
        rlay_fp,
        feat_l,
        cores=None,
        **kwargs):
    #===========================================================================
    # defaults
    #===========================================================================
    if cores is None: 
        cores = multiprocessing.cpu_count()
    #===========================================================================
    # define callers
    #===========================================================================
    def get_chunks(data, n):
        """Yield successive n-sized chunks from a slice-able iterable."""
        for i in range(0, len(data), n):
            yield (data[i:i+n], rlay_fp)

    #chukn the data
    chunks = get_chunks(feat_l, cores)

    # Create a process pool using all cores
    
    with multiprocessing.Pool(cores) as pool:

        # parallel map
        stats_lists = starmap_with_kwargs(pool, _zonal_stats_partial, chunks, repeat(kwargs))
        #stats_lists = pool.starmap(_zonal_stats_partial, chunks)

    # flatten to a single list
    stats = list(itertools.chain(*stats_lists))

    assert len(stats) == len(feat_l)
    
    return stats


 
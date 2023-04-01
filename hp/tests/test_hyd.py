'''
Created on Apr. 1, 2023

@author: cefect
'''


import pytest, tempfile, datetime, os, copy, math
import numpy as np
import rasterio as rio

from hp.hyd import (
    write_poly_to_rlay, write_rlay_to_poly, get_inun_ar,
    )
from definitions import src_dir

#===============================================================================
# toy data
#===============================================================================
from hp.tests.tools.rasters import get_rlay_fp
test_dir = os.path.join(src_dir, 'hp/tests/data')
assert os.path.exists(test_dir), test_dir

from hp.tests.data.toy_rasters import proj_ar_d

#get a polygon from this
wse1_fp = get_rlay_fp(proj_ar_d['wse13'], 'wse13')
inun_fp = write_rlay_to_poly(wse1_fp, dkey='WSE') #uniform
#inun_ar = get_inun_ar(wse1_mar, 'WSE')


@pytest.mark.parametrize('poly_fp',
                         [inun_fp]
                         )
@pytest.mark.parametrize('rlay_ref', [wse1_fp])                        
def test_write_poly_to_rlay(poly_fp, rlay_ref):
    write_poly_to_rlay(poly_fp, rlay_ref=rlay_ref)
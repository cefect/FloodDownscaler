'''
Rasterio
'''
import os
import numpy as np
import pandas as pd
import numpy.ma as ma
import rasterio
import rasterio.features
import rasterio.warp
from matplotlib.pyplot import imshow
from rasterio.enums import Resampling

fp=r'C:\Users\cefect\Downloads\scratch.tif'

#===============================================================================
# setup matplotlib
#===============================================================================
 
import matplotlib
matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt

#set teh styles
plt.style.use('default')

#font
matplotlib.rc('font', **{
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 8})

for k,v in {
    'axes.titlesize':10,
    'axes.labelsize':10,
    'figure.titlesize':12,
    'figure.autolayout':False,
    'figure.figsize':(10,6),
    'legend.title_fontsize':'large'
    }.items():
        matplotlib.rcParams[k] = v
 
print('loaded matplotlib %s'%matplotlib.__version__)

from hp.plot import plot_rast

#===============================================================================
# upscale a raster
#===============================================================================
upscale_factor = 10

with rasterio.open(fp, 'r') as dataset:
    
    
    #===========================================================================
    # meta
    #===========================================================================
    assert dataset.count==1, 'only setup for single band'
    msk = dataset.read_masks(1)  #read the GDAL RFC 15 mask
    nodata_cnt = (msk==0).sum()
    ndval = dataset.nodata
 
    d = {'name':dataset.name, 'shape':str(dataset.shape),
         'nodata_cnt':nodata_cnt, 'size':np.prod(dataset.shape),
         'crs':dataset.crs, 'ndval':dataset.nodata}
    print('loaded {shape} raster \'{name}\' on {crs} w/  {nodata_cnt}/{size} nulls (ndval={ndval}) '.format(**d))
    
 

    #===========================================================================
    # # resample data to target shape
    #===========================================================================
    out_shape=(dataset.count,int(dataset.height * upscale_factor),int(dataset.width * upscale_factor))
    print('transforming from %s to %s'%(dataset.shape, out_shape))
    data_rsmp = dataset.read(
        out_shape=out_shape,
        resampling=Resampling.bilinear
    )[0]
    
    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data_rsmp.shape[-1]),
        (dataset.height / data_rsmp.shape[-2])
    )
    
    
    #===========================================================================
    # resample nulls
    #===========================================================================
    msk_rsmp = dataset.read_masks(1, 
            out_shape=out_shape,
            resampling=Resampling.nearest, #doesnt bleed
        ) 
    
    #===========================================================================
    # plot_rast(msk_rsmp)
    # plot_rast(data, ndval=dataset.nodata)
    # print(data.shape)
    #===========================================================================
    
#===============================================================================
# coerce nulls onto data
#===============================================================================
#ma.array(data, mask = msk_rsmp)
assert data_rsmp.shape==msk_rsmp.shape
data_rsmp_f1 = np.where(msk_rsmp==0,  dataset.nodata, data_rsmp)

print((data_rsmp==ndval).sum())
print((data_rsmp_f1==ndval).sum())
print((msk_rsmp==0).sum())


#===============================================================================
# plot_rast(msk_rsmp)
# plot_rast(data_rsmp_f1, ndval=ndval)
#===============================================================================
#===============================================================================
# ar1 = np.where(msk_rsmp==0, data, dataset.nodata)
# 
# (data==ndval).sum()
# (ar1==ndval).sum()
# (msk_rsmp==0).sum()
# pd.Series(msk_rsmp.reshape((1, -1))[0]).value_counts()
# 
# s1 = pd.Series(ar1.reshape((1, -1))[0])
# s1.replace({ndval:np.nan}).dropna().hist()
#  
# plot_rast(ar1, ndval=dataset.nodata)
#===============================================================================

    
 
    
#===============================================================================
# #write result
#===============================================================================
ofp = os.path.join(os.path.expanduser('~'), 'scratch_res2.tif')
with rasterio.open(
    ofp,
    'w',
    driver='GTiff',
    height=data_rsmp.shape[0],
    width=data_rsmp.shape[1],
    count=1,
    dtype=data_rsmp.dtype,
    crs=dataset.crs,
    transform=transform,
    nodata=dataset.nodata,
    ) as dst:
    dst.write(data_rsmp_f1, 1, 
              masked=False, #not using numpy.ma
              )
    
    #dst.write_mask(msk_rsmp) #write the resampled mask to a sidecar
    
    
print('wrote to \n    %s'%ofp)
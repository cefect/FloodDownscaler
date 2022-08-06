'''
Rasterio
'''
import os
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

upscale_factor = 2

with rasterio.open(fp) as dataset:

    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * upscale_factor),
            int(dataset.width * upscale_factor)
        ),
        resampling=Resampling.bilinear
    )[0]

    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]),
        (dataset.height / data.shape[-2])
    )
    
    dataset.read_masks(1)
    
    print(data.shape)
    """stopped here
    how are nodata values handled?
    """
    
    #imshow(data)
    
#write result
ofp = os.path.join(os.path.expanduser('~'), 'scratch_res.tif')
with rasterio.open(
    ofp,
    'w',
    driver='GTiff',
    height=data.shape[0],
    width=data.shape[1],
    count=1,
    dtype=data.dtype,
    crs=dataset.crs,
    transform=transform,
    ) as dst:
    dst.write(data, 1)
    
print(ofp)
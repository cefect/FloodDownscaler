'''
Created on Jul. 20, 2021

@author: cefect
'''


import imageio.v3 as iio
import imageio
import datetime, os
print('imageio: %s'%imageio.__version__)

def capture_images( #take a directory of images and convert into a gif animation
        ofp,
        img_dir, #directory with images
        ):
    """WARNING: images need to have the same dimensions... no geospatial support"""
    
    filenames =  [os.path.join(img_dir, e) for e in os.listdir(img_dir) if e.endswith('.tif')]
    #os.chdir(img_dir)
    print('on %i files\n    %s'%(len(filenames), filenames))
    
    """
    imageio.help(name='gif')
    imageio.help(name='tif')
    help(imageio.get_writer)
    """

    #with iio.get_writer(ofp, mode='I', duration=0.5) as writer:
    frames=[]
    for filename in filenames:
        #assert os.path.exists(filename)
        frames.append(iio.imread(filename, plugin='GDAL'))
        #writer.append_data(image)
        #break
            
    iio.imwrite(ofp, frames,
                #plugin='Pillow',
                format_hint=".gif", 
                mode='I', #32bit
                size=(1000,1000),
                )
    print('animation saved to \n    %s'%ofp)
            
            
            
if __name__ =="__main__": 
    
    start =  datetime.datetime.now()
    print('start at %s'%start)


    #build_hmax()
    
    capture_images(
        r'C:\LS\10_OUT\2112_Agg\outs\hrast1\r2\20220427\gif\out.gif',
        r'C:\LS\10_OUT\2112_Agg\outs\hrast1\r2\20220427\temp',
        )

    
    

    
    tdelta = datetime.datetime.now() - start
    print('finished in %s'%tdelta)
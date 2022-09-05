import numpy as np
import scipy.ndimage
from skimage.transform import resize
from skimage.filters import gaussian

from scipy.io import loadmat

def createCropsAndSave(origimg, m, outputdir, num_times = 40):
    """
    the function samples velocity models by cropping randomly 
    rotated and scaled images
    """
    wavespeedlist = []
    if type(origimg)!=list : 
        nimg = 1
        origimg = np.expand_dims(origimg,0)
    else: nimg = len(origimg)

    for k in range(nimg):
        for j in range(num_times):
            print('wavespeed',k,'crop', j)
        # randomly scale
            #scale = np.random.randint(1,7) / 5.0 # scale of [0.25,1]
            scale = 0.08+0.04*np.random.rand()
        # random angle in 22.5 increments
            angle = np.random.randint(4) * 22.5  # in degrees

            M = int(m/scale)  # this is how much we crop before resizing to m
       
        # rotate
            npimg = scipy.ndimage.rotate(origimg[k], angle, cval=1.0, order=4, mode='wrap') # bilinear interp
            #npimg = origimg[k]
            h,w = npimg.shape

        # crop but make sure it is not blank image
            while True:
                xTopLeft = np.random.randint(max(1,w-M)) 
                yTopLeft = np.random.randint(max(1,h-M)) 
                newim = npimg[yTopLeft:yTopLeft+M,xTopLeft:xTopLeft+M]

                if (newim.std()>0.005 and newim.mean()<3.8 and not np.all(newim==0)):
                    npimg = 1.0*newim
                    break
        # resize
            npimg = resize(npimg,(m,m),order=4)
            #/np.random.randint(3,10) 

            wavespeedlist.append(npimg)
        
    np.savez(outputdir,wavespeedlist = wavespeedlist)


if __name__ == '__main__':

    datamat = loadmat('marm1nonsmooth.mat')
    fullmarm = gaussian(datamat['marm1larg'],4)
    databp = loadmat('bp2004.mat')
    fullbp = gaussian(databp['V'],4)/1000
    #seed = np.random.randint(1e5+1)
        
    createCropsAndSave([fullmarm,fullbp], m=256, outputdir = 'mabp4sig_size256cropsM100.npz',num_times=50)


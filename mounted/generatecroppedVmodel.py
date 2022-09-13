import numpy as np
import scipy.ndimage
from skimage.transform import resize
from skimage.filters import gaussian
from scipy.io import loadmat

def createCropsAndSave(v_images, m, outputdir, num_times = 40):
    """
    the function samples velocity models by cropping randomly 
    rotated and scaled images
    """

    wavespeed_list = []
    i = 0
    for img in v_images:
        print("img",i)
        i += 1
        for j in range(num_times):

            scale = 0.08+0.04*np.random.rand() #chose this scaling because performed well
            angle = np.random.randint(4) * 22.5  #in degrees
            M = int(m/scale)  #how much we crop before resizing to m
            npimg = scipy.ndimage.rotate(img, angle, cval=1.0, order=4, mode='wrap') # bilinear interp and rotation
            h,w = npimg.shape

            #crop but make sure it is not blank image
            while True:
                xTopLeft = np.random.randint(max(1,w-M)) 
                yTopLeft = np.random.randint(max(1,h-M)) 
                newim = npimg[yTopLeft:yTopLeft+M,xTopLeft:xTopLeft+M]

                if (newim.std()>0.005 and newim.mean()<3.8 and not np.all(newim==0)):
                    npimg = 1.0*newim
                    break

            wavespeed_list.append(resize(npimg,(m,m),order=4))
        
    np.savez(outputdir,wavespeedlist = wavespeed_list)


if __name__ == '__main__':

    print("start running generatecroppedVmodel.py")

    datamat = loadmat('../data/marm1nonsmooth.mat') #velocity models Marmousi dataset
    fullmarm = gaussian(datamat['marm1larg'],4) #to make smoother
    databp = loadmat('../data/bp2004.mat') #velocity models BP dataset
    fullbp = gaussian(databp['V'],4)/1000 #to make smoother (and different order of magnitude)
        
    createCropsAndSave([fullmarm,fullbp],
                       m=256,
                       outputdir = '../data/mabp4sig_size256cropsM100.npz',
                       num_times=50)

    print("finish running generatecroppedVmodel.py")


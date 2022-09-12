import numpy as np
import scipy.ndimage
from skimage.transform import resize
from skimage.filters import gaussian
from scipy.io import loadmat
from tqdm import tqdm

#TODO: simpify; just use torchs randomcrop and scale?

def createCropsAndSave(img_list, m = 256, outputdir = "mabp4sig_size256cropsM100.npz", num_times = 50):

    """
    the function samples velocity models by cropping randomly 
    rotated and scaled images
    """

    wavespeedlist = []

    for img in img_list: #iterate over all images, and perform random modifications num_times
        for _ in tqdm(range(num_times)):

            #TODO: is that uniformly, or what is the reason for this random scale and rotation?

            #manipulate image
            scale = 0.08 + 0.04 * np.random.rand() #randomly scale
            angle = np.random.randint(4) * 22.5  #random angle in 22.5 increments in degrees
            M = int(m/scale)  #this is how much we crop before resizing to m
            npimg = scipy.ndimage.rotate(img, angle, cval=1.0, order=4, mode='wrap') # bilinear interp, rotate

            #crop
            h, w = npimg.shape
            while True:
                xTopLeft = np.random.randint(max(1,w-M)) 
                yTopLeft = np.random.randint(max(1,h-M)) 
                newim = npimg[yTopLeft:yTopLeft+M,xTopLeft:xTopLeft+M]

                # make sure image is not blank image
                if (newim.std()>0.005 and newim.mean()<3.8 and not np.all(newim==0)):
                    npimg = 1.0*newim
                    break

            npimg = resize(npimg,(m,m),order=4) #resize to mxm
            wavespeedlist.append(npimg)

    np.savez(outputdir, wavespeedlist = wavespeedlist) #store numpy arrays in compressed format


if __name__ == '__main__':

    datamat = loadmat('mounted/marm1nonsmooth.mat') #image saved in .mat file
    fullmarm = gaussian(datamat['marm1larg'],4) #filter operation
    databp = loadmat('mounted/bp2004.mat')
    fullbp = gaussian(databp['V'],4)/1000

    createCropsAndSave([fullmarm,fullbp], num_times = 5)

    # import matplotlib.pyplot as plt
    #
    # plt.plot(fullbp)
    # plt.show()
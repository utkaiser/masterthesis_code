import numpy as np
import scipy.ndimage
from skimage.transform import resize
from skimage.filters import gaussian
from scipy.io import loadmat


def generate_velocity_profile_crop(
        v_images,
        m,
        output_path,
        num_times
):
    '''

    Parameters
    ----------
    v_images : (tensor) full-size velocity profile that needs to be cropped
    m : (int) resolution, usually 128 (*1, *2 or *3)
    output_path : (string) path where to save generated crops
    num_times : (int) number of crops

    Returns
    -------
    sample velocity profiles by cropping randomly rotated and scaled images
    '''

    wavespeed_list = []

    for img in v_images:
        for j in range(num_times):
            scale = 0.08+0.04*np.random.rand()  # chose this scaling because performed well
            angle = np.random.randint(4) * 22.5   # in degrees
            M = int(m/scale)   # how much we crop before resizing to m
            npimg = scipy.ndimage.rotate(img, angle, cval=1.0, order=4, mode='wrap')  # bilinear interp and rotation
            h,w = npimg.shape

            # crop but make sure it is not blank image
            while True:
                xTopLeft = np.random.randint(max(1,w-M))
                yTopLeft = np.random.randint(max(1,h-M))
                newim = npimg[yTopLeft:yTopLeft+M,xTopLeft:xTopLeft+M]

                if newim.std()>0.005 and newim.mean()<3.8 and not np.all(newim == 0):
                    npimg = 1.0*newim
                    break

            wavespeed_list.append(resize(npimg,(m,m),order=4))
        
    np.savez(output_path, wavespeedlist = wavespeed_list)


if __name__ == '__main__':

    datamat = loadmat('../data/velocity_profiles/marm1nonsmooth.mat') #velocity models Marmousi dataset
    fullmarm = gaussian(datamat['marm1larg'],4) #to make smoother
    databp = loadmat('../data/velocity_profiles/bp2004.mat') #velocity models BP dataset
    fullbp = gaussian(databp['V'],4)/1000 #to make smoother (and different order of magnitude)


    config = {
        "../data/velocity_profiles/crops_bp_m_400_128.npz": 128
    }

    for k,v in config.items():
        generate_velocity_profile_crop(
            [fullmarm,fullbp],
            m=v,
            output_path = k,
            num_times=200
        )

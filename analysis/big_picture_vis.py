import sys
sys.path.append("..")
import numpy as np
import torch
from scipy.io import loadmat
from skimage.filters import gaussian
from generate_data.wave_util import crop_center, WaveEnergyField_tensor
from models.model_utils import get_params
from generate_data.wave_propagation import velocity_verlet_tensor
import matplotlib.pyplot as plt

def visualize_big_picture(centers, widths, bc = "absorbing"):

    res = 256 #2422
    tj = 17

    # init condition
    # datamat = loadmat('../data/marm1nonsmooth.mat')  # velocity models Marmousi dataset
    # fullmarm = gaussian(datamat['marm1larg'], 4)  # to make smoother

    a = np.load("../data/crops_bp_m_200_2000.npz")['wavespeedlist']
    res = 0
    for i in range(200):
        res = max(res,np.max(a[i,:,:]))

    print(res)

    # print(max([max(max()) for i in range(200)]))

    # vel = torch.from_numpy(crop_center(fullmarm, res, res))
    # k = 0.1
    # vel[120:180,100:200] = k
    # vel[180:200, 120:180] = k
    # vel[200:210, 150:170] = k
    # vel[180:190, 180:190] = k
    # vel[130:170, 80:100] = k
    #
    # plt.imshow(vel, vmax=4, vmin=2.5)
    # plt.savefig("../analysis/vis_abc3/vel_img.png")
    # #plt.show()
    #
    # param_dict = get_params()
    # x = np.linspace(-1, 1, res)
    # y = np.linspace(-1, 1, res)
    # xx, yy = np.meshgrid(x, y)
    #
    # centers = [(-110,-110),(110,-110),(-110,110),(110,110)]#,(10,10),(10,10),(10,10),(10,10),(10,10),(10,10),(10,10),(10,10),(10,10),(10,10),(10,10)]
    #
    # centers = [(c/256,d/ 256) for c,d in centers]
    # widths = [2000,2000,2000,2000]
    #
    # u0 = torch.from_numpy(np.exp(-widths[0] * ((xx - centers[0][0]) ** 2 + (yy - centers[0][1]) ** 2)))
    #
    # for j in range(1, 3):
    #     u0 += torch.from_numpy(np.exp(-widths[j] * ((xx - centers[j][0]) ** 2 + (yy - centers[j][1]) ** 2)))
    #
    # ut0 = torch.from_numpy(np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)]))
    #
    # ucx, utcx = u0.clone(), ut0.clone()
    # ucx, utcx = ucx.squeeze(), utcx.squeeze()
    # for j in range(1, tj):
    #
    #     #ucx += torch.from_numpy(np.exp(-widths[7+j] * ((xx - centers[5+j][0]) ** 2 + (yy - centers[5+j][1]) ** 2)))*.3
    #
    #     ucx, utcx = velocity_verlet_tensor(
    #         ucx.unsqueeze(dim=0), utcx.unsqueeze(dim=0), vel.unsqueeze(dim=0), param_dict["f_delta_x"],
    #         param_dict["f_delta_t"], param_dict["delta_t_star"], number=1, boundary_c=bc, tj=j
    #     )
    #
    #     ucx, utcx = ucx.squeeze(), utcx.squeeze()
    #
    #     # plt.imshow(
    #     #     WaveEnergyField_tensor(ucx, utcx, vel, param_dict["f_delta_t"]) * param_dict["f_delta_t"] * param_dict[
    #     #         "f_delta_t"])
    #     # plt.axis('off')
    #     # plt.show()
    #     # plt.savefig("vis_abc/"+bc+str(j)+".png")
    # print("hallo")

if __name__ == '__main__':

    # centers, widths = np.random.rand(50, 2) * 2. - 1, 500 + (np.random.randn(50)**2) * 1000
    #
    # #visualize_big_picture(centers=centers, widths=widths, bc="periodic")
    # visualize_big_picture(centers=centers, widths=widths, bc="absorbing")

    visualize_big_picture(None, None)



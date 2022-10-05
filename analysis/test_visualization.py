import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    a = velf = np.load("../data/crops_bp_m_200_256.npz")
    vellist = velf['wavespeedlist']
    for i in range(100):
        plt.imshow(vellist[i+100])
        plt.show()







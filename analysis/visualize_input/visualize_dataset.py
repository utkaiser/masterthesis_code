import sys
sys.path.append("..")
from generate_data.initial_conditions import get_velocity_dict
import matplotlib.pyplot as plt
import numpy as np

def vis_velocities():

    input_path = "../../data/crops_bp_m_200_128.npz"
    velocities = get_velocity_dict(128,10,input_path)

    # choose how many from which
    for key, value in velocities.items():
        if key == "bp_m":
            velocities[key] = np.concatenate([value[:8],value[-8:]], axis=0)
        else:
            velocities[key] = np.expand_dims(value[0], axis=0)

    velocity_tensor = np.concatenate(list(velocities.values()), axis=0)
    print(velocity_tensor.shape)

    fig = plt.figure(figsize=(8, 8))

    for i in range(velocity_tensor.shape[0]):
        vel = velocity_tensor[i]
        a = fig.add_subplot(5, 4, i+1)
        plt.imshow(vel)
        a.set_aspect('equal')
        plt.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=-.4, hspace=.1)
    plt.savefig("visualized_velocity_profiles.pdf")
    plt.close(fig)

if __name__ == '__main__':
    vis_velocities()




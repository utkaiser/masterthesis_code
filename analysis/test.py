import matplotlib.pyplot as plt
import random
from scipy.stats import truncnorm


def get_truncated_normal(mean, sd, low, upp):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

if __name__ == '__main__':

    n_snaps = 10
    add = 1
    plt.figure(figsize=(80,50))
    for epoch in range(20):
        if (epoch + 1) % 3 == 0: add +=1
        print("-"*20, epoch)
        for i in range(50):
            for inpt_idx in random.choices(range(n_snaps-2), k=n_snaps):
                label_range = round(get_truncated_normal(mean=min(n_snaps-1,inpt_idx+add), sd=1, low=inpt_idx+1, upp=n_snaps-1).rvs())
                if label_range == inpt_idx:
                    print("a")
                if label_range == n_snaps:
                    print("aa")






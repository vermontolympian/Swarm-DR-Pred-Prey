import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def norm(array, axis=0):
    return np.sqrt(np.sum(np.power(array, 2), axis=axis))



if __name__ == '__main__':
    with open('prey-speed-DRTest6.csv', 'r', newline='') as csvfile:
        # Initialize CSV Reader
        fieldnames = ['M', 'c', 'd', 'e', 'vel1', 'vel2', 'vel3', 'vel4', 'ang1', 'ang2', 'ang3', 'ang4']

        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        i = 0
        for row in reader:
            if i > 0:
                vel = [row['vel1'], row['vel2'], row['vel3'], row['vel4']]
                ang = [row['ang1'], row['ang2'], row['ang3'], row['ang4']]
            i += 1

        n = [500, 1000, 1500, 2000]
        fig = plt.scatter(vel, ang)
        # plt.xlim(0.1, 0.03)
        plt.ylim(-3.2, 3.2)
        for i, txt in enumerate(n):
            plt.annotate(txt, (vel[i], ang[i]))
        # Plot heatmaps
        plt.show()


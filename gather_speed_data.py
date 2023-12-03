import csv
import time
import datetime
import numpy as np
from model_numba import run_model
import matplotlib.pyplot as plt

sim_params = {
    "T": 2000,
    "dt": 0.05,
    "W": 4,
    "H": 4
}

vis_params = {
    "fps": 24,
    "ticks_per_frame": 6,
    "show_animation": False,
    "export_animation": False,
    "animation_path": "models/animation.mp4",
    "animation_codec": "h264"
}

def norm(array, axis=0):
    return np.sqrt(np.sum(np.power(array, 2), axis=axis))



if __name__ == "__main__":
    with (open('prey-speed-DRTest6.csv', 'w', newline='') as csvfile):
        fieldnames = ['M', 'c', 'd', 'e', 'caught']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        permutations = []
        for M in [2, 3, 4, 5]:
            for d, e in [(0.5, 0.4), (0.5, 0.5), (0.5, 0.6), (0.5, 0.7), (0.5, 0.8), (0.5, 0.9), (0.5, 1.0), (0.5, 1.1)]:
                for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    permutations.append((M, d, e, c))

        # for M in [2]:
        #     for d, e in [(0.5, 0.4)]:
        #         for c in [0.1, 1]:
        #             permutations.append((M, d, e, c))

        avg_time = 0
        run_start = time.time()
        for i, p in enumerate(permutations):
            M = p[0]
            d = p[1]
            e = p[2]
            c = p[3]
            start_time = time.time()
            np.random.seed(0)
            vis_params['animation_path'] = f"media/DRTest6/Replication_M{M}_c{c}_d{d}_e{e}.mp4"
            caught, all_prey_v = run_model(
                a=1,        # Prey-Prey Linear Long-Range Attraction Gain
                b=0.2,      # Prey-Predator Repulsion Strength
                c=c,      # Pred-Prey Attraction Strength
                d=d,      # Pred-Pred Exponential Short Range Repulsion
                e=e,        # Pred-Pred Linear Long-Range Attraction Gain
                p=3,        # Power law for Predator-Prey Interactions
                N=400,
                M=M,
                sim_params=sim_params, vis_params=vis_params
            )
            if caught:
                print("Predator Caught Prey!")

            writer.writerow({
                'M': M,
                'c': c,
                'd': d,
                'e': e,
                'caught': int(caught)
            })

            vel = norm(all_prey_v[:, 0, :], axis=1)
            ang = np.arctan2(all_prey_v[:, 0, 1], all_prey_v[:, 0, 0])

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(vel, ang)
            plt.title(f"M{M}_c{c}_d{d}_e{e}")
            ax.set_xscale("log")
            plt.xlabel('Linear Velocity')
            plt.ylabel('Angular Position (Rad)')
            plt.savefig(f"SpeedGraphsLog/M{M}_c{c}_d{d}_e{e}.png", dpi=200)
            # plt.show()

            this_time = int(time.time() - start_time)
            avg_time = int((avg_time*i/(i+1)) + (this_time/(i+1)))
            time_elapsed = int(time.time()-run_start)
            time_remaining = int((len(permutations)-(i+1))*avg_time)
            print(f"Timer: ")
            print(f"\tthis iter      = {datetime.timedelta(seconds=this_time)},")
            print(f"\tavg iter       = {datetime.timedelta(seconds=avg_time)},")
            print(f"\ttime elapsed   = {datetime.timedelta(seconds=time_elapsed)},")
            print(f"\ttime remaining = {datetime.timedelta(seconds=time_remaining)}")
            print("")
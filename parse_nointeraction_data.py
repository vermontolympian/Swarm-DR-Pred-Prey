import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

parse_behavior = True
parse_catching = True

if __name__ == '__main__':
    with open('data/MultiPredatorNoInteraction_Characterization_kr.025.csv', 'r', newline='') as csvfile:
        # Intialize CSV Reader
        fieldnames = ['M', 'c', 'd', 'e', 'caught', 'behavior']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        # Values for each subplot
        m_values = np.array([1])#[5,4,3,2,1])
        m_labels = [str(m) for m in m_values]
        # X Values/Labels
        c_values = np.array([0.15, 0.4, 0.8, 1.5, 2.5])
        c_labels = [str(c) for c in c_values]

        # Values for Heatmap
        behavior_labels = ["Escaping", "Stable Confusion",
                           "Unstable Confusion", "Coordinated Circling", "Chaotic Confusion"]
        behaviors = np.empty((m_values.size, c_values.size,),
                             dtype=np.int16)
        catching = np.empty((m_values.size, c_values.size,),
                            dtype=np.int16)
        # Parse Data for heatmaps
        i = 0
        for row in reader:
            if i > 0:
                m_ind = np.where(m_values == int(row['M']))[0]
                c_ind = np.where(c_values == float(row['c']))[0]
                # print(int(row['M']), float(row['c']), int(
                #     row['caught']), int(row['behavior']))
                if parse_behavior:
                    behaviors[m_ind,
                            c_ind,
                            ] = int(row['behavior'])
                if parse_catching:
                    catching[m_ind,
                            c_ind,
                            ] = int(row['caught'])
            i += 1
        print(behaviors)
        print(catching)
        # Plot heatmaps
        if parse_behavior:
            fig = plt.figure(1)
            ax = fig.gca()
            fig.suptitle("Flocking Behavior w.r.t M Predators and C Gains")
            min_behavior = np.min(behaviors)
            max_behavior = np.max(behaviors)+1
            cmap = sns.color_palette(palette="Spectral", 
                                     n_colors=int(max_behavior-min_behavior))
            sns.heatmap(behaviors, xticklabels=c_labels,
                        yticklabels=m_labels, ax=ax, cbar=False, square=True, cmap=cmap)
            ax.set_ylabel("m")
            ax.set_xlabel("c")

            handles = []
            for i in range(int(max_behavior-min_behavior)):
                b_handle = mpatches.Patch(
                    color=cmap[i], label=behavior_labels[i])
                handles.append(b_handle)
            fig.legend(handles=handles, loc='lower center', ncols=5)

        if parse_catching:
            fig2 = plt.figure(2)
            ax2 = fig2.gca()
            fig2.suptitle("Catching Behavior w.r.t M Predators and C Gains")
            cmap2 = sns.color_palette("blend:#C00,#0C0", n_colors=2)
            sns.heatmap(catching, xticklabels=c_labels,
                        yticklabels=m_labels, ax=ax2, cbar=False, square=True, cmap=cmap2)
            ax2.set_ylabel("m")
            ax2.set_xlabel("c")

            escape_patch = mpatches.Patch(
                color=cmap2[0], label="Did Not Catch")
            catch_patch = mpatches.Patch(color=cmap2[1], label="Did Catch")
            handles = [escape_patch, catch_patch]
            fig2.legend(handles=handles, loc='lower center', ncols=2)

        plt.show()

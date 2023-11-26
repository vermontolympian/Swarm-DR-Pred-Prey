import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

parse_behavior = True
parse_catching = True

if __name__ == '__main__':
    with open('predator_caught_prey-DRTest5.csv', 'r', newline='') as csvfile:
        # Intialize CSV Reader
        fieldnames = ['M', 'c', 'd', 'e']
        if parse_catching:
            fieldnames.append('caught')
        if parse_behavior:
            fieldnames.append('behavior')
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        # Values for each subplot
        m_values = np.array([2, 3, 4, 5])
        m_labels = np.array(['2', '3', '4', '5'])
        # X Values/Labels
        c_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        c_labels = [str(c) for c in c_values]
        # Y Values/Labels
        de_values = np.array([0.5/0.4, 0.5/0.5, 0.5/0.6, 0.5/0.7,
                              0.5/0.8, 0.5/0.9, 0.5/1.0, 0.5/1.1])
        de_labels = ['0.5, 0.4', '0.5, 0.5', '0.5, 0.6', '0.5, 0.7',
                     '0.5, 0.8', '0.5, 0.9', '0.5, 1.0', '0.5, 1.1']
        # Values for Heatmap
        behavior_labels = ["Escaping", "Stable",
                           "Unstable", "Circling", "Chaotic"]
        behaviors = np.zeros((m_values.size, de_values.size, c_values.size,),
                             dtype=np.int16)
        catching = np.zeros((m_values.size, de_values.size, c_values.size,),
                            dtype=np.int16)
        # Parse Data for heatmaps
        i = 0
        for row in reader:
            if i > 0:
                m_ind = np.where(m_values == int(row['M']))[0]
                c_ind = np.where(c_values == float(row['c']))[0]
                de_ind = np.where(de_values == float(
                    row['d'])/float(row['e']))[0]
                if parse_behavior:
                    behaviors[m_ind,
                                de_ind,
                                c_ind,
                                ] = int(row['behavior'])
                if parse_catching:
                    catching[m_ind,
                                de_ind,
                                c_ind,
                                ] = int(row['caught'])
            i += 1
        # Plot heatmaps
        if parse_behavior:
            fig, axs = plt.subplots(2, 2)#, sharex='col', sharey='row')
            fig.suptitle("Flocking Behavior for M values")
            min_behavior = np.min(behaviors)
            max_behavior = np.max(behaviors)+1
            cmap = sns.color_palette(
                palette="Spectral", n_colors=int(max_behavior-min_behavior))
            for i, ax in enumerate(axs.flatten().tolist()):
                ax.set_title(f"M={m_labels[i]}", fontdict={"size": 10})
                sns.heatmap(behaviors[i, :, :], xticklabels=c_labels,
                            yticklabels=de_labels, ax=ax, cbar=False, square=True, cmap=cmap)
                if i % axs.shape[1] == 0:
                    ax.set_ylabel("de")
                if int(i/axs.shape[1]) == axs.shape[0]-1:
                    ax.set_xlabel("c")

            handles = []
            for i in range(int(max_behavior-min_behavior)):
                b_handle = mpatches.Patch(
                    color=cmap[i], label=behavior_labels[i])
                handles.append(b_handle)
            fig.legend(handles=handles, loc='lower center', ncols=5)

        if parse_catching:
            fig2, axs2 = plt.subplots(2, 2)#, sharex='col', sharey='row')
            fig2.suptitle("Catching Behavior for M values")
            cmap2 = sns.color_palette("blend:#C00,#0C0", n_colors=2)
            for i, ax in enumerate(axs2.flatten().tolist()):
                ax.set_title(f"M={m_labels[i]}", fontdict={"size": 10})
                sns.heatmap(catching[i, :, :], xticklabels=c_labels,
                            yticklabels=de_labels, ax=ax, cbar=False, square=True, cmap=cmap2)
                if i % axs.shape[1] == 0:
                    ax.set_ylabel("de")
                if int(i/axs.shape[1]) == axs.shape[0]-1:
                    ax.set_xlabel("c")

            escape_patch = mpatches.Patch(
                color=cmap2[0], label="Did Not Catch")
            catch_patch = mpatches.Patch(color=cmap2[1], label="Did Catch")
            handles = [escape_patch, catch_patch]
            fig2.legend(handles=handles, loc='lower center', ncols=2)

        plt.show()

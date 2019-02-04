import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp


colours = ["red", "blue", "green", "cyan", "orange", "magenta"]
# names = ["eef_discrete", "eef_euclid", "eef_euclid_dist", "embed_discrete", "eef_embed", "embed_embed"]
names = ["eef_discrete", "embed_discrete", "eef_euclid", "embed_embed", "eef_embed"]
# names = ["embed_discrete", "embed_embed"]


def plot_data(filename, outdir, mus, vars, names):
    print(outdir)
    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(1, 1, 1)
    t = np.linspace(1, 1, num=len(mus[0]))
    ax.plot(t, lw=1, label='target mean', color='black', ls='--', linewidth=2)

    for mu, var, name, c in zip(mus, vars, names, colours):
        sigma = var

        if name == "eef_discrete":
            label = "EE_D"
        elif name == "eef_euclid":
            label = "EE_Eu"
        elif name == "eef_euclid_dist":
            label = "EE_EuD"
        elif name == "eef_embed":
            label = "EE_Em"
        elif name == "embed_discrete":
            label = "Em_D"
        elif name == "embed_embed":
            label = "Em_Em"

        # the n_sigma sigma upper and lower analytic population bounds
        lower_bound3 = mu - 3 * sigma
        upper_bound3 = mu + 3 * sigma

        n_sigma = 1
        lower_bound_n = mu - n_sigma * sigma
        upper_bound_n = mu + n_sigma * sigma

        ax.plot(mu, label=label, color=c, linewidth=2)
        ax.fill_between(np.arange(len(mus[0])), lower_bound_n, upper_bound_n, facecolor=c, alpha=0.2)

    if filename == "all":
        ax.set_title("All", fontsize=20, weight="bold")
    elif name == "eef_discrete":
        ax.set_title("State:Position, Reward: Discrete", fontsize=20, weight="bold")
    elif name == "eef_euclid":
        ax.set_title("State:Position, Reward: Position Cosine", fontsize=20, weight="bold")
    elif name == "eef_euclid_dist":
        ax.set_title("State:Position, Reward: Euclidean Dist", fontsize=20, weight="bold")
    elif name == "eef_embed":
        ax.set_title("State:Pose, Reward: Embed Cosine", fontsize=20, weight="bold")
    elif name == "embed_discrete":
        ax.set_title("State:Embed, Reward: Discrete", fontsize=20, weight="bold")
    elif name == "embed_embed":
        ax.set_title("State:Embed, Reward: Embed Cosine", fontsize=20, weight="bold")


    plt.legend(loc="lower right", prop={'size': 20})
    ax.set_ylabel('Enf-of-episode Reward', fontsize=20, weight="bold")
    ax.set_xlabel('# episodes', fontsize=20, weight="bold")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, len(t))

    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14)

    for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

    ax.grid()

    if filename is not None:
        plt.savefig(osp.join(outdir, filename + "_stats.png"), bbox_inches="tight")
    else:
        plt.savefig(osp.join(outdir, name + "_stats.png"), bbox_inches="tight")
    plt.show()
    plt.close()

def plot(folder_name=None, trials=None, outdir=None, names=[]):

    episode_len = 15
    val_episodes = 10
    mus = []
    all = []
    diffs = []

    for index, trial in enumerate(trials):

        all.append([])

        data = np.load(osp.join(folder_name, trial, "training_stats.npz"))["stats"]
        diffs += list(np.load(osp.join(folder_name, trial, "training_stats.npz"))["diffs"][-val_episodes:])

        no_episodes = len(data) - val_episodes
        
        # no_episodes = 30
        train_data = data[:no_episodes]
        valid_data = data[no_episodes:]
        
        for i, x in enumerate(train_data):
            all[index].append(x[-1])

    mus = np.mean(all, axis=0)
    vars = np.std(all, axis=0)

    # print("Diffs", diffs)
    print("Means diffs", np.mean(diffs, axis=0))
    print("Std diffs", np.std(diffs, axis=0))

    plot_data(None, outdir, [mus], [vars], names)

    return mus, vars

if __name__ == "__main__":

    all_mus = []
    all_vars = []

    if os.listdir("images") == []:
        next_stats = 0
    else:
        next_stats = int(sorted([int(x) for x in os.listdir("images")])[-1]) + 1
        os.mkdir(osp.join("images", str(next_stats)))

    for name in names:
        no_last_trials = 10
        trials = sorted(os.listdir(osp.join("result_mock_embed", name, "right_arm_right")))[-no_last_trials:]

        prefix = osp.join("result_mock_embed", name, "right_arm_right")

        print(name)
        mus, vars = plot(folder_name=prefix, trials=trials, outdir=osp.join("images", str(next_stats)), names=[name])

        all_mus.append(mus)
        all_vars.append(vars)

    print("All")
    plot_data("all", osp.join("images", str(next_stats)), all_mus, np.zeros(np.array(all_vars).shape), names)

import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *

plt.rcParams.update({
    "font.family": "serif",        # 类似论文的衬线字体
})


def plot_L2_var(npz_path="errors.npz", field:str =None, name: str=None, save_path=None):
    key = "L2_{}".format(field)
    data = np.load(npz_path)
    L2_T = data[key]          # (num_epochs,)
    epochs = np.arange(1, len(L2_T) + 1)

    plt.figure(figsize=(7.2, 4.8), dpi=300, constrained_layout=True)
    plt.plot(epochs, L2_T, marker="o", markersize=3)

    plt.xlabel("Epoch")
    plt.ylabel(r"$L2$ Error (Total)")
    plt.title(r"Training $L2$ Error over Epochs")
    plt.tight_layout()
    if save_path is not None:
        save_path = os.path.join(save_path, f"{name}_{key}.png")
        save_fig(save_path)
    else:
        plt.show()

def plot_L2_each(npz_path="errors.npz", name=None, save_path=None):
    data = np.load(npz_path)
    L2_each = data["L2_each"]    # (num_epochs, 7)
    num_epochs, num_comp = L2_each.shape

    epochs = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(7.2, 4.8), dpi=300, constrained_layout=True)

    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]

    for i in range(num_comp):
        plt.plot(
            epochs,
            L2_each[:, i],
            linestyle=linestyles[i % len(linestyles)],
            marker="o",
            markersize=2,
            label=fr"Step {i+1}"
        )

    plt.xlabel("Epoch")
    plt.ylabel(r"$L2$ Error (Per Step)")
    plt.title(r"Per-Step $L2$ Error over Epochs", pad=25)

    # 图例放在外侧右边，更学术风（内容多时不挡图）
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=7, fontsize=7, frameon=True)

    plt.tight_layout()

    if save_path is not None:
        save_path = os.path.join(save_path, f"{name}_L2_each.png")
        save_fig(save_path)
    else:
        plt.show()


PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
load_path = os.path.join(PATH, r"Result/data/Test_result.npz")
save_path = os.path.join(PATH, r"Result/fig")
save_name = "Test"

# plot_L2_var(npz_path=load_path, field='T', name=save_name, save_path=save_path)
plot_L2_each(npz_path=load_path, name=save_name, save_path=save_path)



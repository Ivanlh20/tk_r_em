# Copyright 2026 Ivan Lobato / NeuralSoftX
# SPDX-License-Identifier: GPL-3.0-only
"""Restore a single experimental HRSTEM image.

Author: Ivan Lobato
Email: ivan.lobato@neuralsoftx.com
"""
import os
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib.pyplot as plt
from tk_r_em import load_network, load_hrstem_exp_test_data


def main():
    # select one of: sfr_hrsem, sfr_lrsem, sfr_hrstem, sfr_lrstem, sfr_hrtem, sfr_lrtem
    net_name = 'sfr_hrstem'

    x = load_hrstem_exp_test_data('sgl_exp_hrstem')
    net = load_network(net_name)
    net.summary()

    t_0 = time.perf_counter()
    y_p = net.predict(x)
    t_e = time.perf_counter() - t_0

    print(f'[{net_name}] {x.shape[0]}x{x.shape[1]} image  |  {t_e:.3f} s')

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for ax, (img, label) in zip(axs, [
        (x, "Experimental"),
        (y_p, "Restored"),
    ]):
        ax.imshow(img, cmap='gray')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=12)

    fig.subplots_adjust(wspace=0.05)
    plt.savefig(f"restored_{net_name}_single.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()

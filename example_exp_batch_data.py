# Copyright 2026 Ivan Lobato / NeuralSoftX
# SPDX-License-Identifier: Apache-2.0
"""
Example 2: restore experimental HRSTEM data (batch of small images).

Loads bundled experimental HRSTEM images, runs batch inference,
and plots a side-by-side comparison: experimental | restored.

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
    batch_size = 8

    x = load_hrstem_exp_test_data('exp_hrstem')
    net = load_network(net_name)
    net.summary()

    t_0 = time.perf_counter()
    y_p = net.predict(x, batch_size=batch_size)
    t_e = time.perf_counter() - t_0

    n = x.shape[0]
    print(f'[{net_name}] {n} images  |  {t_e:.3f} s  |  '
          f'{1e3*t_e/n:.2f} ms/image  |  {n/t_e:.1f} images/s')

    n_cols = x.shape[0]
    fig, axs = plt.subplots(2, n_cols, figsize=(2.5 * n_cols, 5))

    for k in range(n_cols):
        for row, (img, label) in enumerate([
            (x[k, :, :, 0], "Experimental"),
            (y_p[k, :, :, 0], "Restored"),
        ]):
            axs[row][k].imshow(img.squeeze(), cmap='gray')
            axs[row][k].set_xticks([]); axs[row][k].set_yticks([])
            if k == 0:
                axs[row][k].set_ylabel(label, fontsize=12)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(f"restored_{net_name}_exp.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()

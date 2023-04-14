"""
tk_r_em network suites designed to restore different modalities of electron microscopy data

Author: Ivan Lobato
Email: Ivanlh20@gmail.com
"""

import os
import matplotlib

# Check if running on remote SSH and use appropriate backend for matplotlib
remote_ssh = "SSH_CONNECTION" in os.environ
matplotlib.use('Agg' if remote_ssh else 'TkAgg')
import matplotlib.pyplot as plt

def fcn_set_gpu_id(gpu_visible_devices: str = "0") -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_visible_devices

fcn_set_gpu_id("0")

from tk_r_em import load_network, load_sim_test_data

def fcn_inference():
    """
    Perform inference on test data using a pre-trained model and visualize the results.
    """
    # select one of the available networks from [sfr_hrsem, sfr_lrsem, sfr_hrstem, sfr_lrstem, sfr_hrtem, sfr_lrtem]
    net_name = 'sfr_hrstem'
    
    # load its corresponding data
    x, y = load_sim_test_data(net_name)

    # load its corresponding model
    r_em_nn = load_network(net_name)
    r_em_nn.summary()

    n_data = x.shape[0]
    batch_size = 8

    # run inference
    y_p = r_em_nn.predict(x, batch_size)

    fig, axs = plt.subplots(3, n_data, figsize=(48, 6))

    for ik in range(n_data):
        x_ik = x[ik, :, :, 0].squeeze()
        y_p_ik = y_p[ik, :, :, 0].squeeze()
        y_ik = y[ik, :, :, 0].squeeze()

        ir = 0
        axs[ir][ik].imshow(x_ik, cmap='viridis')
        axs[ir][ik].set_xticks([])
        axs[ir][ik].set_yticks([])
        axs[ir][ik].grid(False)
        
        if ik == 0:
            axs[ir][ik].set_ylabel(f"Detected {net_name} image", fontsize=14, )

        ir = 1
        axs[ir][ik].imshow(y_p_ik, cmap='viridis')
        axs[ir][ik].set_xticks([])
        axs[ir][ik].set_yticks([])
        axs[ir][ik].grid(False)

        if ik == 0:
            axs[ir][ik].set_ylabel(f"Restored {net_name} image", fontsize=14)
        
        ir = 2
        axs[ir][ik].imshow(y_ik, cmap='viridis')
        axs[ir][ik].set_xticks([])
        axs[ir][ik].set_yticks([])
        axs[ir][ik].grid(False)

        if ik == 0:
            axs[ir][ik].set_ylabel(f"Ground truth {net_name} image", fontsize=14)

    fig.subplots_adjust(hspace=2, wspace=10)
    fig.tight_layout()
    
    if remote_ssh:
        plt.savefig(f"restored_{net_name}.png", format='png')
    else:
        fig.show()

    print('Done')

if __name__ == '__main__':
    fcn_inference()
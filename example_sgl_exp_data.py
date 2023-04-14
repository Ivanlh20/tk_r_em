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

from tk_r_em import load_network, load_hrstem_exp_test_data

def fcn_inference():
    """
    Perform inference on test data using a pre-trained model and visualize the results.
    """
    # select one of the available networks from [sfr_hrsem, sfr_lrsem, sfr_hrstem, sfr_lrstem, sfr_hrtem, sfr_lrtem]
    net_name = 'sfr_hrstem'
    
    # load experimental hrstem data
    x = load_hrstem_exp_test_data('sgl_exp_hrstem')
        
    # load its corresponding model
    r_em_nn = load_network(net_name)
    r_em_nn.summary()

    # run inference
    y_p = r_em_nn.predict_patch_based(x, patch_size=256, stride=128, batch_size=16)

    fig, axs = plt.subplots(1, 2, figsize=(48, 6))
    ir = 0
    axs[ir].imshow(x, cmap='hot')
    axs[ir].set_xticks([])
    axs[ir].set_yticks([])
    axs[ir].grid(False)
    axs[ir].set_title(f"Experimental {net_name} image", fontsize=14, )

    ir = 1
    axs[ir].imshow(y_p, cmap='hot')
    axs[ir].set_xticks([])
    axs[ir].set_yticks([])
    axs[ir].grid(False)
    axs[ir].set_title(f"Restored {net_name} image", fontsize=14)

    fig.subplots_adjust(hspace=2, wspace=10)
    fig.tight_layout()
    
    if remote_ssh:
        plt.savefig(f"restored_{net_name}.png", format='png')
    else:
        fig.show()

    print('Done')

if __name__ == '__main__':
    fcn_inference()
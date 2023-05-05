"""
r_em network suites designed to restore different modalities of electron microscopy data

Author: Ivan Lobato
Email: Ivanlh20@gmail.com
"""
import os
import pathlib
from typing import Tuple

import h5py
import numpy as np
import tensorflow as tf

def expand_dimensions(x):
    if x.ndim == 2:
        return np.expand_dims(x, axis=(0, 3))
    elif x.ndim == 3 and x.shape[-1] != 1:
        return np.expand_dims(x, axis=3)
    else:
        return x

def add_extra_row_or_column(x):
    if x.shape[1] % 2 == 1:
        v_mean = x.mean(axis=(1, 2), keepdims=True)
        v_mean_tiled = np.tile(v_mean, (1, 1, x.shape[2], 1))
        x = np.concatenate((x, v_mean_tiled), axis=1)

    if x.shape[2] % 2 == 1:
        v_mean = x.mean(axis=(1, 2), keepdims=True)
        v_mean_tiled = np.tile(v_mean, (1, x.shape[1], 1, 1))
        x = np.concatenate((x, v_mean_tiled), axis=2)

    return x

def add_extra_row_or_column_patch_based(x):
    if x.shape[0] % 2 == 1:
        v_mean = x.mean(axis=(0, 1), keepdims=True)
        v_mean_tiled = np.tile(v_mean, (1, x.shape[1]))
        x = np.concatenate((x, v_mean_tiled), axis=0)

    if x.shape[1] % 2 == 1:
        v_mean = x.mean(axis=(0, 1), keepdims=True)
        v_mean_tiled = np.tile(v_mean, (x.shape[0], 1))
        x = np.concatenate((x, v_mean_tiled), axis=1)

    return x

def remove_extra_row_or_column(x, x_i_sh):
    if x_i_sh != x.shape:
        return x[:, :x_i_sh[1], :x_i_sh[2], :]
    else:
        return x

def remove_extra_row_or_column_patch_based(x, x_i_sh):
    if x_i_sh != x.shape:
        return x[:x_i_sh[0], :x_i_sh[1]]
    else:
        return x
    
def adjust_output_dimensions(x, x_i_shape):
    ndim = len(x_i_shape)
    if ndim == 2:
        return x.squeeze()
    elif ndim == 3:
        if x_i_shape[-1] == 1:
            return x.squeeze(axis=0)
        else:
            return x.squeeze(axis=-1)  
    else:
        return x

def get_centered_range(n, patch_size, stride):
    patch_size_half = patch_size // 2
    if patch_size_half == n-patch_size_half:
        return np.array([patch_size_half])

    p = np.arange(patch_size_half, n-patch_size_half, stride)
    if p[-1] + patch_size_half < n:
        p = np.append(p, n - patch_size_half)
    return p

def get_range(im_shape, patch_size, strides):
    py = get_centered_range(im_shape[0], patch_size[0], strides[0])
    px = get_centered_range(im_shape[1], patch_size[1], strides[1])

    for iy in py:
        for ix in px:
            yield slice(iy - patch_size[0] // 2, iy + patch_size[0] // 2), slice(ix - patch_size[1] // 2, ix + patch_size[1] // 2)

def process_prediction(data, x_r, count_map, window, ib, sy, sx):
    for ik in range(ib):
        x_r_ik = data[ik, ..., 0].squeeze() * window
        count_map[sy[ik], sx[ik]] += window
        x_r[sy[ik], sx[ik]] += x_r_ik

def butterworth_window(shape, cutoff_radius_ftr, order):
    assert len(shape) == 2, "Shape must be a tuple of length 2 (height, width)"
    assert 0 < cutoff_radius_ftr <= 0.5, "Cutoff frequency must be in the range (0, 0.5]"

    def butterworth_1d(length, cutoff_radius_ftr, order):
        n = np.arange(-length//2, length-length//2)
        window = 1 / (1 + (n / (cutoff_radius_ftr * length)) ** (2 * order))
        return window

    window_y = butterworth_1d(shape[0], cutoff_radius_ftr, order)
    window_x = butterworth_1d(shape[1], cutoff_radius_ftr, order)
    window = np.outer(window_y, window_x)

    return window

class Model(tf.keras.Model):
    def __init__(self, model_path):
        super(Model, self).__init__()
        self.base_model = tf.keras.models.load_model(model_path, compile=False)
        self.base_model.compile()
        
    def call(self, inputs, training=None, mask=None):
        return self.base_model(inputs, training=training, mask=mask)
        
    def summary(self):
        return self.base_model.summary()
    
    def predict(self, x, batch_size=16, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        x_i_sh = x.shape

        # Expanding dimensions based on the input shape
        x = expand_dimensions(x)

        # Converting to float32 if necessary
        x = x.astype(np.float32)

        x_i_sh_e = x.shape

        # Adding extra row or column if necessary
        x = add_extra_row_or_column(x)

        batch_size = min(batch_size, x.shape[0])

        # Model prediction
        x = self.base_model.predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)

        # Removing extra row or column if added
        x = remove_extra_row_or_column(x, x_i_sh_e)

        # Adjusting output dimensions to match input dimensions
        return adjust_output_dimensions(x, x_i_sh)

    def predict_patch_based(self, x, patch_size=None, stride=None, batch_size=16):
        if patch_size is None:
            return self.predict(x, batch_size=batch_size)

        x = x.squeeze().astype(np.float32)

        x_i_sh_e = x.shape
        
        # Adding extra row or column if necessary
        x = add_extra_row_or_column_patch_based(x)
        
        patch_size = max(patch_size, 128)
        patch_size = (min(patch_size, x.shape[0]), min(patch_size, x.shape[1]))
        
        # Adjust the stride to have an overlap between patches
        overlap = (patch_size[0]//2, patch_size[1]//2)
        if stride is None:
            stride = overlap
        else:
            stride = (min(stride, overlap[0]), min(stride, overlap[1]))

        batch_size = max(batch_size, 4)

        data = np.zeros((batch_size, *patch_size, 1), dtype=np.float32)
        sy = [slice(0) for _ in range(batch_size)]
        sx = [slice(0) for _ in range(batch_size)]

        x_r = np.zeros(x.shape, dtype=np.float32)
        count_map = np.zeros(x.shape, dtype=np.float32)
        
        window = butterworth_window(patch_size, 0.33, 4)
            
        ib = 0
        for s_iy, s_ix in get_range(x.shape, patch_size, stride):
            if ib < batch_size:
                data[ib, ..., 0] = x[s_iy, s_ix]
                sy[ib] = s_iy
                sx[ib] = s_ix
                ib += 1

                if ib == batch_size:
                    data = self.base_model.predict(data, batch_size=batch_size)
                    process_prediction(data, x_r, count_map, window, ib, sy, sx)
                    ib = 0

        if ib != batch_size:
            data = self.base_model.predict(data[:ib, ...], batch_size=batch_size)
            process_prediction(data, x_r, count_map, window, ib, sy, sx)
            
        # Normalize the denoised image using the count_map
        x_r /= count_map
        
        # Removing extra row or column if added
        x = remove_extra_row_or_column_patch_based(x, x_i_sh_e)

        return x_r

def load_network(model_name: str = 'sfr_hrstem'):
    """
    Load r_em neural network model.

    :param model_name: A string representing the name of the model.
    :return: A tensorflow.keras.Model object.
    """
    if os.path.isdir(model_name):
        model_path = pathlib.Path(model_name).resolve()
    else: 
        model_name = model_name.lower()
        model_path = pathlib.Path(__file__).resolve().parent / 'models' / model_name

    model = Model(model_path)

    return model

def load_sim_test_data(file_name: str = 'sfr_hrstem') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data for r_em neural network.

    :param model_name: A string representing the name of the model.
    :return: A tuple containing two numpy arrays representing the input (x) and output (y) data.
    """
    if os.path.isfile(file_name):
        path = pathlib.Path(file_name).resolve()
    else:
        file_name = file_name.lower()
        path = pathlib.Path(__file__).resolve().parent / 'test_data' / f'{file_name}.h5'


    with h5py.File(path, 'r') as h5file:
        x = np.asarray(h5file['x'][:], dtype=np.float32).transpose(0, 3, 2, 1)
        y = np.asarray(h5file['y'][:], dtype=np.float32).transpose(0, 3, 2, 1)
    
    return x, y

def load_hrstem_exp_test_data(file_name: str = 'exp_hrstem') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data for r_em neural network.

    :param model_name: A string representing the name of the model.
    :return: A tuple containing two numpy arrays representing the input (x) and output (y) data.
    """

    if os.path.isfile(file_name):
        path = pathlib.Path(file_name).resolve()
    else:
        file_name = file_name.lower()
        path = pathlib.Path(__file__).resolve().parent / 'test_data' / f'{file_name}.h5'

    with h5py.File(path, 'r') as f:
        x = f['x'][:]
        if x.ndim == 4:
            x = np.asarray(x, dtype=np.float32).transpose(0, 3, 2, 1)
        else:
            x = np.asarray(x, dtype=np.float32).transpose(1, 0)
    
    return x
# Copyright 2026 Ivan Lobato / NeuralSoftX
# SPDX-License-Identifier: GPL-3.0-only
"""ONNX-based inference for r_em electron microscopy image restoration networks.

Author: Ivan Lobato
Email: ivan.lobato@neuralsoftx.com
"""

import os
import pathlib

import h5py
import numpy as np
try:
    import onnxruntime as ort
except ImportError as _e:
    raise ImportError(
        "tk_r_em requires ONNX Runtime, which is not installed. Pick one of:\n"
        "    pip install tk_r_em[cpu]        # portable CPU-only (Linux/Windows/macOS)\n"
        "    pip install tk_r_em[gpu]        # NVIDIA GPU (Linux/Windows)\n"
        "    pip install tk_r_em[directml]   # any DirectX 12 GPU on Windows\n"
        "Do NOT install both `onnxruntime` and `onnxruntime-gpu` in the same "
        "environment — they conflict and the CPU wheel silently wins at import."
    ) from _e


# ---------------------------------------------------------------------------
# Preload NVIDIA CUDA wheels so that ORT can dlopen them at session creation.
# Only meaningful on CUDA builds; skip on the CPU-only wheel to avoid a
# noisy "onnxruntime is not built with CUDA 12.x support" warning that the
# ORT C++ layer prints on a stock CPU build.
# ---------------------------------------------------------------------------
if ort.get_device() != 'CPU':
    import warnings
    try:
        ort.preload_dlls()
    except Exception as _dll_err:
        warnings.warn(
            f"ort.preload_dlls() failed: {_dll_err}. "
            "CUDA sessions may still work but could be slower.",
            stacklevel=1,
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = pathlib.Path(__file__).resolve().parent / 'models'
TEST_DATA_DIR = pathlib.Path(__file__).resolve().parent / 'test_data'


# ---------------------------------------------------------------------------
# Provider selection: CUDA → DirectML → CPU
# ---------------------------------------------------------------------------
_AVAILABLE_PROVIDERS = ort.get_available_providers()
_HAS_CUDA_BUILD = 'CUDAExecutionProvider' in _AVAILABLE_PROVIDERS
_HAS_DML_BUILD = 'DmlExecutionProvider' in _AVAILABLE_PROVIDERS

if _HAS_CUDA_BUILD:
    _PROVIDER_PREFS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
elif _HAS_DML_BUILD:
    _PROVIDER_PREFS = [('DmlExecutionProvider', {'device_id': 0}),
                       'CPUExecutionProvider']
else:
    _PROVIDER_PREFS = ['CPUExecutionProvider']

_REQUESTED_DML = _HAS_DML_BUILD and not _HAS_CUDA_BUILD

_SESSIONS = {}
_DEVICE = None
_DEVICE_NAME = None


def _get_session(path):
    """Lazy-load an ONNX session by path and cache it for the process lifetime."""
    global _DEVICE, _DEVICE_NAME
    path = str(path)
    if path not in _SESSIONS:
        so = ort.SessionOptions()
        so.log_severity_level = 3  # error-only
        if _REQUESTED_DML:
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(path, sess_options=so,
                                    providers=_PROVIDER_PREFS)
        if _DEVICE is None:
            chosen = sess.get_providers()
            if 'CUDAExecutionProvider' in chosen:
                _DEVICE, _DEVICE_NAME = 'cuda', 'cuda'
            elif 'DmlExecutionProvider' in chosen:
                _DEVICE, _DEVICE_NAME = 'dml', 'directml'
            else:
                _DEVICE, _DEVICE_NAME = 'cpu', 'cpu'
        _SESSIONS[path] = sess
    return _SESSIONS[path]


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _to_batch(x):
    """Normalise (H,W), (N,H,W), or (N,H,W,1) input to (N,H,W,1)."""
    if x.ndim == 2:
        return x[np.newaxis, ..., np.newaxis]
    if x.ndim == 3 and x.shape[-1] != 1:
        return x[..., np.newaxis]
    return x


def _from_batch(x, original_shape):
    """Squeeze (N,H,W,1) output back to match the original input rank."""
    ndim = len(original_shape)
    if ndim == 2:
        return x.squeeze()
    if ndim == 3:
        return x.squeeze(axis=0) if original_shape[-1] == 1 else x.squeeze(axis=-1)
    return x


def _pad_to_even(x):
    """Pad spatial dims to even sizes with the spatial mean. Works on 2D and 4D."""
    spatial_axes = (1, 2) if x.ndim == 4 else (0, 1)
    needs_pad = [ax for ax in spatial_axes if x.shape[ax] % 2 == 1]
    if not needs_pad:
        return x
    mean_val = x.mean(axis=spatial_axes, keepdims=True)
    for ax in needs_pad:
        shape = list(x.shape)
        shape[ax] = 1
        x = np.concatenate((x, np.broadcast_to(mean_val, shape)), axis=ax)
    return x


def _crop_to_shape(x, target_shape):
    """Crop x back to target_shape (undo padding)."""
    if x.shape == target_shape:
        return x
    return x[tuple(slice(s) for s in target_shape)]


# ---------------------------------------------------------------------------
# Patch-based helpers
# ---------------------------------------------------------------------------

def _centered_range(n, patch_size, stride):
    half = patch_size // 2
    if half == n - half:
        return np.array([half])
    p = np.arange(half, n - half, stride)
    if p[-1] + half < n:
        p = np.append(p, n - half)
    return p


def _patch_slices(im_shape, patch_size, stride):
    """Yield (slice_y, slice_x) for each patch position."""
    py = _centered_range(im_shape[0], patch_size[0], stride[0])
    px = _centered_range(im_shape[1], patch_size[1], stride[1])
    half_h, half_w = patch_size[0] // 2, patch_size[1] // 2
    for iy in py:
        for ix in px:
            yield slice(iy - half_h, iy + half_h), slice(ix - half_w, ix + half_w)


def _accumulate_patches(predictions, output, count_map, window, n, sy, sx):
    """Add windowed predictions into the running output and count map."""
    for k in range(n):
        output[sy[k], sx[k]] += predictions[k, :, :, 0] * window
        count_map[sy[k], sx[k]] += window


def _butterworth_window(shape, cutoff=0.33, order=4):
    """2D separable Butterworth low-pass window."""
    def _bw1d(length):
        n = np.arange(-length // 2, length - length // 2)
        return 1.0 / (1.0 + (n / (cutoff * length)) ** (2 * order))
    return np.outer(_bw1d(shape[0]), _bw1d(shape[1]))


# ---------------------------------------------------------------------------
# Network wrapper
# ---------------------------------------------------------------------------

class _onnx_network:
    """ONNX-based inference wrapper for a single r_em restoration network."""

    def __init__(self, model_path):
        self._path = str(model_path)
        self._session = _get_session(self._path)
        self._input_name = self._session.get_inputs()[0].name

    def summary(self):
        """Print model info including the resolved execution device."""
        inp = self._session.get_inputs()[0]
        out = self._session.get_outputs()[0]
        size_mb = pathlib.Path(self._path).stat().st_size / 1024 / 1024
        print(f'r_em network: {pathlib.Path(self._path).stem}')
        print(f'  Input:  {inp.name}  {inp.shape}  {inp.type}')
        print(f'  Output: {out.name}  {out.shape}  {out.type}')
        print(f'  Device: {_DEVICE_NAME}')
        print(f'  Size:   {size_mb:.1f} MB')

    def _run(self, x):
        """Run the ONNX session on a single batch."""
        return self._session.run(
            None, {self._input_name: np.ascontiguousarray(x, dtype=np.float32)}
        )[0]

    def predict(self, x, batch_size=16):
        """Whole-image inference with auto-padding to even H/W."""
        original_shape = x.shape
        x = _to_batch(x).astype(np.float32, copy=False)
        shape_before_pad = x.shape
        x = _pad_to_even(x)

        n = x.shape[0]
        batch_size = min(batch_size, n)

        if n <= batch_size:
            x = self._run(x)
        else:
            x = np.concatenate(
                [self._run(x[i:i + batch_size]) for i in range(0, n, batch_size)]
            )

        x = _crop_to_shape(x, shape_before_pad)
        return _from_batch(x, original_shape)

    def predict_patch_based(self, x, patch_size=None, stride=None, batch_size=16):
        """Patch-based inference with Butterworth-windowed blending."""
        if patch_size is None:
            return self.predict(x, batch_size=batch_size)

        x = x.squeeze().astype(np.float32)
        original_shape = x.shape
        x = _pad_to_even(x)

        patch_size = max(patch_size, 128)
        patch_size = (min(patch_size, x.shape[0]), min(patch_size, x.shape[1]))

        overlap = (patch_size[0] // 2, patch_size[1] // 2)
        if stride is None:
            stride = overlap
        else:
            stride = (min(stride, overlap[0]), min(stride, overlap[1]))

        batch_size = max(batch_size, 4)
        window = _butterworth_window(patch_size)

        output = np.zeros(x.shape, dtype=np.float32)
        count_map = np.zeros(x.shape, dtype=np.float32)
        batch_buf = np.zeros((batch_size, *patch_size, 1), dtype=np.float32)
        sy, sx = [], []

        ib = 0
        for s_iy, s_ix in _patch_slices(x.shape, patch_size, stride):
            batch_buf[ib, :, :, 0] = x[s_iy, s_ix]
            sy.append(s_iy)
            sx.append(s_ix)
            ib += 1

            if ib == batch_size:
                preds = self._run(batch_buf)
                _accumulate_patches(preds, output, count_map, window, ib, sy, sx)
                sy.clear()
                sx.clear()
                ib = 0

        if ib > 0:
            preds = self._run(batch_buf[:ib])
            _accumulate_patches(preds, output, count_map, window, ib, sy, sx)

        output /= count_map
        return _crop_to_shape(output, original_shape)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VALID_TAGS = ('sfr_hrsem', 'sfr_lrsem', 'sfr_hrstem',
               'sfr_lrstem', 'sfr_hrtem', 'sfr_lrtem')


def load_network(model_name: str = 'sfr_hrstem'):
    """Load an r_em restoration network by tag or file path."""
    if os.path.isfile(model_name):
        return _onnx_network(pathlib.Path(model_name).resolve())
    model_name = model_name.lower()
    model_path = MODELS_DIR / f'{model_name}.onnx'
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No model '{model_name}'. Valid tags: {', '.join(_VALID_TAGS)}"
        )
    return _onnx_network(model_path)


def load_sim_test_data(file_name: str = 'sfr_hrstem') -> tuple[np.ndarray, np.ndarray]:
    """Load simulated test data (x, y) for a given modality tag."""
    if os.path.isfile(file_name):
        path = pathlib.Path(file_name).resolve()
    else:
        file_name = file_name.lower()
        path = TEST_DATA_DIR / f'{file_name}.h5'

    with h5py.File(path, 'r') as h5file:
        x = np.asarray(h5file['x'][:], dtype=np.float32).transpose(0, 3, 2, 1)
        y = np.asarray(h5file['y'][:], dtype=np.float32).transpose(0, 3, 2, 1)

    return x, y


def load_hrstem_exp_test_data(file_name: str = 'exp_hrstem'):
    """Load experimental HRSTEM test data."""
    if os.path.isfile(file_name):
        path = pathlib.Path(file_name).resolve()
    else:
        file_name = file_name.lower()
        path = TEST_DATA_DIR / f'{file_name}.h5'

    with h5py.File(path, 'r') as f:
        x = f['x'][:]
        if x.ndim == 4:
            x = np.asarray(x, dtype=np.float32).transpose(0, 3, 2, 1)
        else:
            x = np.asarray(x, dtype=np.float32).transpose(1, 0)

    return x

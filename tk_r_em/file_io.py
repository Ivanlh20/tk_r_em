# Copyright 2026 Ivan Lobato / NeuralSoftX
# SPDX-License-Identifier: GPL-3.0-only
"""Multi-format image loading for electron microscopy data.

Author: Ivan Lobato
Email: ivan.lobato@neuralsoftx.com
"""
import os
import tempfile

import numpy as np
from PIL import Image


_EM_EXTENSIONS = {'.ser', '.dm3', '.dm4', '.emd'}
_PIL_EXTENSIONS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}


def _load_pil(file_obj):
    """Load image via PIL, convert to float32 greyscale."""
    img = Image.open(file_obj).convert("F")
    return np.array(img, dtype=np.float32)


def _load_em_format(file_obj, ext):
    """Load SER/DM3/DM4/EMD via rosettasciio, writing to temp file first."""
    from rsciio.tia import file_reader as _ser_reader
    from rsciio.digitalmicrograph import file_reader as _dm_reader
    from rsciio.emd import file_reader as _emd_reader

    readers = {
        '.ser': _ser_reader,
        '.dm3': _dm_reader,
        '.dm4': _dm_reader,
        '.emd': _emd_reader,
    }
    reader = readers[ext]

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        tmp.write(file_obj.read())
        tmp.close()
        signals = reader(tmp.name)
    finally:
        os.unlink(tmp.name)

    data = signals[0]['data']
    if data.ndim > 2:
        data = data[0]
    return data.astype(np.float32)


def load_image(uploaded_file):
    """Load an uploaded file as a 2D float32 numpy array."""
    name = uploaded_file.name.lower()
    ext = os.path.splitext(name)[1]

    if ext in _PIL_EXTENSIONS:
        data = _load_pil(uploaded_file)
    elif ext in _EM_EXTENSIONS:
        data = _load_em_format(uploaded_file, ext)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if data.ndim > 2:
        data = data[0]

    return data

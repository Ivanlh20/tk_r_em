# Copyright 2026 Ivan Lobato / NeuralSoftX
# SPDX-License-Identifier: Apache-2.0
"""Multi-format image loading for electron microscopy data.

Author: Ivan Lobato
Email: ivan.lobato@neuralsoftx.com
"""
import os
import tempfile

import numpy as np
from PIL import Image


_EM_EXTENSIONS = {'.ser', '.dm3', '.dm4'}
_PIL_EXTENSIONS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}


def _load_pil(file_obj):
    """Load image via PIL, convert to float32 greyscale."""
    img = Image.open(file_obj).convert("F")
    return np.array(img, dtype=np.float32)


def _load_em_format(file_obj, ext):
    """Load SER/DM3/DM4 via rosettasciio, writing to temp file first."""
    import rosettasciio

    readers = {
        '.ser': rosettasciio.ser.file_reader,
        '.dm3': rosettasciio.dm3.file_reader,
        '.dm4': rosettasciio.dm4.file_reader,
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

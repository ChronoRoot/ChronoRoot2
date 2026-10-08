"""Multi-page TIFF stacks for per-plant Seg / SegMulti masks.

Sequential BigTIFF writer: each page is deflated once, then appended at
EOF with an 8-byte IFD patch. No OpenCV. No reread of earlier pages.
"""

import csv
import os
import struct
import zlib

import numpy as np
from PIL import Image

SEG_TIFF_NAME = "Seg.tif"
SEGMULTI_TIFF_NAME = "SegMulti.tif"

_TIFF_SHORT = 3
_TIFF_LONG = 4
_TIFF_LONG8 = 16
_COMPRESSION_ADOBE_DEFLATE = 8
_ZLIB_LEVEL = 6


def seg_tiff_path(images_dir):
    return os.path.join(images_dir, SEG_TIFF_NAME)


def segmulti_tiff_path(images_dir):
    return os.path.join(images_dir, SEGMULTI_TIFF_NAME)


def mask_stack_paths(result_dir):
    """Return Seg / SegMulti locations and whether they are TIFF stacks."""
    images = os.path.join(result_dir, "Images")
    seg_tif = seg_tiff_path(images)
    multi_tif = segmulti_tiff_path(images)
    if os.path.isfile(seg_tif) or os.path.isfile(multi_tif):
        return {
            "kind": "tiff",
            "images": images,
            "seg": seg_tif,
            "seg_multi": multi_tif,
        }
    return {
        "kind": "png",
        "images": images,
        "seg": os.path.join(images, "Seg"),
        "seg_multi": os.path.join(images, "SegMulti"),
    }


def tiff_n_frames(path):
    if not path or not os.path.isfile(path):
        return 0
    try:
        with Image.open(path) as image:
            return int(getattr(image, "n_frames", 1) or 1)
    except OSError:
        return 0


def read_tiff_page(path, idx, bgr=False):
    """Decode a single TIFF page.

    Grayscale pages are HxW uint8. Color pages are RGB unless bgr=True,
    which swaps to OpenCV's BGR order without importing cv2.
    """
    if not path or not os.path.isfile(path):
        return None
    with Image.open(path) as image:
        n_frames = int(getattr(image, "n_frames", 1) or 1)
        if idx < 0:
            idx += n_frames
        if idx < 0 or idx >= n_frames:
            return None
        image.seek(idx)
        image.load()
        arr = np.array(image)
    return _normalize_page(arr, bgr=bgr)


def result_frame_names(result_dir):
    """Basenames from Results_raw.csv FileName, in analysis order."""
    csv_path = os.path.join(result_dir, "Results_raw.csv")
    if not os.path.isfile(csv_path):
        return []
    names = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "FileName" not in reader.fieldnames:
            return []
        for row in reader:
            name = (row.get("FileName") or "").strip()
            if name:
                names.append(os.path.basename(name))
    return names


def graph_path_for_name(graphs_dir, filename):
    stem = os.path.basename(filename or "")
    lower = stem.lower()
    if lower.endswith(".png"):
        stem = stem[:-4]
    elif lower.endswith(".tif") or lower.endswith(".tiff"):
        stem = os.path.splitext(stem)[0]
    return os.path.join(graphs_dir, stem + ".xml.gz")


def _page_bytes(array, rgb=False):
    arr = np.ascontiguousarray(array)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if rgb:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        arr = np.ascontiguousarray(arr[:, :, ::-1])
        return arr, 3
    if arr.ndim == 3:
        arr = (
            0.114 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.299 * arr[:, :, 2]
        ).astype(np.uint8)
    return np.ascontiguousarray(arr), 1


def _bigtiff_tags(width, height, samples, data_offset, byte_count):
    photometric = 2 if samples == 3 else 1
    if samples == 3:
        bits_count = 3
        bits_value = struct.unpack("<Q", struct.pack("<HHH", 8, 8, 8) + b"\x00\x00")[0]
    else:
        bits_count = 1
        bits_value = 8
    tags = [
        (256, _TIFF_LONG, 1, int(width)),
        (257, _TIFF_LONG, 1, int(height)),
        (258, _TIFF_SHORT, bits_count, bits_value),
        (259, _TIFF_SHORT, 1, _COMPRESSION_ADOBE_DEFLATE),
        (262, _TIFF_SHORT, 1, photometric),
        (273, _TIFF_LONG8, 1, int(data_offset)),
        (277, _TIFF_SHORT, 1, int(samples)),
        (278, _TIFF_LONG, 1, int(height)),
        (279, _TIFF_LONG8, 1, int(byte_count)),
    ]
    if samples == 3:
        tags.append((284, _TIFF_SHORT, 1, 1))
    tags.sort(key=lambda item: item[0])
    return tags


class SequentialTiffWriter:
    """Append one deflated BigTIFF page per frame. Disk work stays O(page)."""

    def __init__(self, path, rgb=False):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.path = path
        self.rgb = bool(rgb)
        self._f = open(path, "w+b")
        self._f.write(b"II")
        self._f.write(struct.pack("<H", 43))
        self._f.write(struct.pack("<H", 8))
        self._f.write(struct.pack("<H", 0))
        self._next_ifd_field = self._f.tell()
        self._f.write(struct.pack("<Q", 0))

    def append(self, array):
        if self._f is None:
            raise RuntimeError(f"TIFF writer already closed: {self.path}")
        arr, samples = _page_bytes(array, rgb=self.rgb)
        height, width = arr.shape[:2]
        payload = zlib.compress(arr.tobytes(), _ZLIB_LEVEL)
        self._align(8)
        data_offset = self._f.tell()
        self._f.write(payload)
        self._align(8)
        ifd_offset = self._f.tell()
        self._patch_u64(self._next_ifd_field, ifd_offset)
        tags = _bigtiff_tags(width, height, samples, data_offset, len(payload))
        self._f.write(struct.pack("<Q", len(tags)))
        for tag, typ, count, value in tags:
            self._f.write(struct.pack("<HHQQ", tag, typ, count, value))
        self._next_ifd_field = self._f.tell()
        self._f.write(struct.pack("<Q", 0))

    def close(self):
        handle = self._f
        self._f = None
        if handle is None:
            return
        try:
            handle.flush()
            handle.close()
        except OSError:
            pass

    def _align(self, nbytes):
        pos = self._f.tell()
        pad = (nbytes - (pos % nbytes)) % nbytes
        if pad:
            self._f.write(b"\x00" * pad)

    def _patch_u64(self, offset, value):
        here = self._f.tell()
        self._f.seek(offset)
        self._f.write(struct.pack("<Q", value))
        self._f.seek(here)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class TiffStackWriter:
    """Append one numpy frame at a time to a multi-page TIFF."""

    def __init__(self, path, rgb=False, compression=None):
        self.path = path
        self.rgb = bool(rgb)
        self._tf = SequentialTiffWriter(path, rgb=self.rgb)

    def append(self, array):
        if self._tf is None:
            raise RuntimeError(f"TIFF writer already closed: {self.path}")
        self._tf.append(array)

    def close(self):
        writer = self._tf
        self._tf = None
        if writer is not None:
            writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class TiffStackReader:
    """Keep one TIFF file handle open and decode pages on demand."""

    def __init__(self, path, bgr=True):
        self.path = path
        self.bgr = bool(bgr)
        self._im = Image.open(path)
        self._n = int(getattr(self._im, "n_frames", 1) or 1)

    def __len__(self):
        return self._n

    def read(self, idx):
        if self._im is None:
            return None
        if idx < 0:
            idx += self._n
        if idx < 0 or idx >= self._n:
            return None
        self._im.seek(idx)
        self._im.load()
        return _normalize_page(np.array(self._im), bgr=self.bgr)

    def close(self):
        image = self._im
        self._im = None
        if image is not None:
            image.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class PlantMaskStacks:
    """Seg.tif + SegMulti.tif writers for one plant run."""

    def __init__(self, images_dir):
        self.seg = TiffStackWriter(seg_tiff_path(images_dir), rgb=False)
        self.multi = TiffStackWriter(segmulti_tiff_path(images_dir), rgb=True)

    def close(self):
        self.seg.close()
        self.multi.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def load_seg_frame(seg_source, idx):
    """Load one frame from a TiffStackReader or a list of image paths."""
    if seg_source is None:
        return None
    if isinstance(seg_source, TiffStackReader):
        return seg_source.read(idx)
    if not seg_source:
        return None
    try:
        length = len(seg_source)
    except TypeError:
        return None
    if idx < 0 or idx >= length:
        return None
    path = seg_source[idx]
    if not path:
        return None
    if str(path).lower().endswith((".tif", ".tiff")):
        return read_tiff_page(path, 0, bgr=True)
    return _read_image_file(path, bgr=True)


def overlay_on_rgb(img, seg):
    """Paint labeled SegMulti pixels onto an RGB image. No OpenCV."""
    if img is None or seg is None:
        return img
    if seg.shape[:2] != img.shape[:2]:
        return img
    out = img.copy()
    if seg.ndim == 3:
        color = seg[..., :3]
        labeled = np.any(color > 0, axis=-1)
        if np.any(labeled):
            out[labeled] = color[labeled]
        return out
    labeled = seg > 0
    if np.any(labeled):
        out[labeled] = (255, 255, 0)
    return out


def _normalize_page(arr, bgr=False):
    if arr.ndim == 3 and arr.shape[2] >= 3:
        if bgr:
            return _swap_rb(arr[..., :3])
        return np.ascontiguousarray(arr[..., :3])
    if arr.ndim == 3:
        return np.ascontiguousarray(arr[:, :, 0])
    return arr


def _swap_rb(arr):
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 2]
    out[..., 1] = arr[..., 1]
    out[..., 2] = arr[..., 0]
    return out


def _read_image_file(path, bgr=True):
    if not path or not os.path.isfile(path):
        return None
    with Image.open(path) as image:
        if image.mode in ("L", "I;16", "I"):
            return np.array(image.convert("L"))
        arr = np.array(image.convert("RGB"))
    if bgr:
        return _swap_rb(arr)
    return arr

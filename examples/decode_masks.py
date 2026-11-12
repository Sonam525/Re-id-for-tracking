"""Read one masks_<clip>.json.gz file. Needs NumPy only.

    python examples/decode_masks.py                    # built-in check, no data needed
    python examples/decode_masks.py masks_<clip>.json.gz

A mask is {"size": [height, width], "counts": [...]}: uncompressed COCO run-length
data in column-major order, starting with a run of background. pycocotools reads it too:

    from pycocotools import mask as mask_utils
    binary = mask_utils.decode(mask_utils.frPyObjects(rle, *rle["size"]))
"""
import gzip
import json
import sys

import numpy as np


def rle_decode(rle):
    """Return an h x w boolean array."""
    h, w = rle["size"]
    counts = np.asarray(rle["counts"], dtype=np.int64)
    values = np.zeros(len(counts), dtype=bool)
    values[1::2] = True
    return np.repeat(values, counts).reshape((h, w), order="F")


def check():
    rle = {"size": [2, 3], "counts": [1, 2, 3]}   # column 0 = 0,1; column 1 = 1,0; column 2 = 0,0
    expected = np.array([[0, 1, 0], [1, 0, 0]], dtype=bool)
    assert (rle_decode(rle) == expected).all()
    print("built-in check passed")


def summary(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        clip = json.load(f)
    frames = clip["frames"]
    print(len(frames), "frames")
    first = frames[0]
    print("frame_idx", first["frame_idx"], "t_sec", first["t_sec"])
    for slot, rle in sorted(first["masks"].items()):
        # Slots are tracker slots, not calves. relabel_week_<day>.json maps them to calves.
        print("slot", slot, "no mask" if rle is None else "%d pixels" % rle_decode(rle).sum())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        summary(sys.argv[1])
    else:
        check()

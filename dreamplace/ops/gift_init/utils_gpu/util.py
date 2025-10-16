import re
import time
import json
import csv
import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import numpy as np
import statistics
from scipy import stats
from pathlib import Path
from scipy.sparse import csc_matrix, lil_matrix, save_npz, load_npz, csgraph, linalg

import logging
logger = logging.getLogger(__name__)

def make_dir(path):
    if os.path.isdir(path):
        print(path, ' dir exists')
    else:
        os.makedirs(path)
        print(path, 'is created')


def find_fixed_point_def(file):
    io_id = []
    io_pos = []
    with open(file, 'r') as f:
        info = f.read()
        totalCellNumber = int(re.search(r'COMPONENTS\s(\d+)\s;', info).group(1))

        # read PIN(IO Pad) info
        PINSRegex = re.compile(r'pins\s+(\d+)', re.IGNORECASE)
        totalPinNumber = int(re.search(PINSRegex, info).group(1)) - 1  # remove clk pin

        PINInfo = info[info.find('PINS'):info.find('END PINS')]
        PINList = re.split(r';', PINInfo)
        PINList.pop(0)
        PINList.pop(-1)

        for i in range(totalPinNumber):
            io_id.append(i + totalCellNumber)
            pos_info = PINList[i].split('\n')[3]
            io_pos.append([int(pos_info.split()[3]), int(pos_info.split()[4])])
    io_pos = np.array(io_pos)

    return totalCellNumber, totalPinNumber, io_id, io_pos


def placement_region(fixed_pos, xl, yl, xh, yh):
    x_min = xl 
    x_max = xh 
    y_min = yl 
    y_max = yh 
    if len(fixed_pos): 
        xf = fixed_pos[:, 0]
        yf = fixed_pos[:, 1]
        x_min = min(np.min(xf), x_min)
        x_max = max(np.max(xf), x_max)
        y_min = min(np.min(yf), y_min)
        y_max = max(np.max(yf), y_max)
    logger.info('placement region: (%g, %g, %g, %g)', x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max, y_max


def _sample_with_risk(risk_map, movable_num, xl, yl, xh, yh):
    """Sample locations biased towards low-risk bins."""
    h, w = risk_map.shape
    risk = risk_map.astype(np.float64)
    max_val = np.max(risk)
    if max_val > 0:
        risk = risk / max_val
    weights = np.exp(-5.0 * risk)
    prob = weights.reshape(-1)
    prob /= np.sum(prob)
    flat_idx = np.random.choice(h * w, size=movable_num, p=prob)
    bin_y = flat_idx // w
    bin_x = flat_idx % w
    bin_size_x = (xh - xl) / w
    bin_size_y = (yh - yl) / h
    jitter_x = np.random.rand(movable_num)
    jitter_y = np.random.rand(movable_num)
    xs = xl + (bin_x + jitter_x) * bin_size_x
    ys = yl + (bin_y + jitter_y) * bin_size_y
    return np.stack([xs, ys], axis=1)


def generate_initial_locations(
    fixed_cell_location, movable_num, xl, yl, xh, yh, scale, risk_map=None
):
    x_min, y_min, x_max, y_max = placement_region(fixed_cell_location, xl, yl, xh, yh)
    xcenter = (x_max - x_min) / 2 + x_min
    ycenter = (y_max - y_min) / 2 + y_min

    if risk_map is not None and risk_map.size:
        sampled = _sample_with_risk(
            risk_map, int(movable_num), x_min, y_min, x_max, y_max
        )
        if abs(scale - 1.0) > 1e-6:
            sampled[:, 0] = ((sampled[:, 0] - xcenter) * scale) + xcenter
            sampled[:, 1] = ((sampled[:, 1] - ycenter) * scale) + ycenter
        return sampled.astype(np.float32)

    sampled = np.random.rand(int(movable_num), 2)
    sampled[:, 0] = ((sampled[:, 0] - 0.5) * (x_max - x_min) * scale) + xcenter
    sampled[:, 1] = ((sampled[:, 1] - 0.5) * (y_max - y_min) * scale) + ycenter
    return sampled.astype(np.float32)

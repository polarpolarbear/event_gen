import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from typing import BinaryIO, Optional, Union

import numpy as np
from numpy.lib import recfunctions

events_struct = np.dtype(
    [("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)]
)

class Events(object):
    """
    Temporal Difference events.
    data: a NumPy Record Array with the following named fields
        x: pixel x coordinate, unsigned 16bit int
        y: pixel y coordinate, unsigned 16bit int
        p: polarity value, boolean. False=off, True=on
        ts: timestamp in microseconds, unsigned 64bit int
    width: The width of the frame. Default = 304.
    height: The height of the frame. Default = 240.
    """
    def __init__(self, num_events, width=304, height=240):
        """num_spikes: number of events this instance will initially contain"""
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.uint16), ('ts', np.uint64)], shape=(num_events))
        self.width = width
        self.height = height

def read_event(filename):
    """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
    f = open(filename, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7 * 1 #bit
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    #Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    #Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    td = Events(td_indices.size, 34, 34)
    td.data.x = all_x[td_indices]
    td.width = td.data.x.max() + 1
    td.data.y = all_y[td_indices]
    td.height = td.data.y.max() + 1
    td.data.ts = all_ts[td_indices]
    td.data.p = all_p[td_indices]

    #further processing for event generation tasks
    #saccades1 = td.data[np.logical_and(0 < td.data['ts'], td.data['ts'] < 100000)] #select the first saccades
    saccades1 = td.data[np.logical_and(0 < td.data['ts'], td.data['ts'] < (105000))] #select the first saccades
    saccades1 = np.array(saccades1.tolist(), dtype=float) #transfer dtype
    
    #saccades1[:,3] = saccades1[:,3] - 0 #normalize time
    saccades1[:,3] = saccades1[:,3] / (105000)   
    
    saccades1[:,0] = saccades1[:,0]/33 #normalize coordinate
    saccades1[:,1] = saccades1[:,1]/33
    #saccades1[:,2][saccades1[:,2]==0] = -1 #convert polarity 0 to -1

    #[x,y,t,p]
    saccades2 = np.array(saccades1.tolist(), dtype=float) #transfer dtype
    saccades2[:,2] = saccades1[:,3]
    saccades2[:,3] = saccades1[:,2]
    return saccades2


def make_structured_array(*args, dtype=events_struct):
    """Make a structured array given a variable number of argument values.

    Parameters:
        *args: Values in the form of nested lists or tuples or numpy arrays.
               Every except the first argument can be of a primitive data type like int or float.

    Returns:
        struct_arr: numpy structured array with the shape of the first argument
    """
    assert not isinstance(
        args[-1], np.dtype
    ), "The `dtype` must be provided as a keyword argument."
    names = dtype.names
    assert len(args) == len(names)
    struct_arr = np.empty_like(args[0], dtype=dtype)
    for arg, name in zip(args, names):
        struct_arr[name] = arg
    return struct_arr

def read_mnist_file(
    bin_file: Union[str, BinaryIO], dtype: np.dtype, is_stream: bool = False):
    """Reads the events contained in N-MNIST/N-CALTECH101 datasets.

    Code adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py
    """
    if is_stream:
        raw_data = np.frombuffer(bin_file.read(), dtype=np.uint8).astype(np.uint32)
    else:
        with open(bin_file, "rb") as fp:
            raw_data = np.fromfile(fp, dtype=np.uint8).astype(np.uint32)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    # Process time stamp overflow events
    time_increment = 2**13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    xytp = make_structured_array(
        all_x[td_indices],
        all_y[td_indices],
        all_ts[td_indices],
        all_p[td_indices],
        dtype=dtype,
    )

    return xytp


def read_mnist_file_by_event(
    bin_file: Union[str, BinaryIO], dtype: np.dtype, is_stream: bool = False, max_n_events=int):
    """Reads the events contained in N-MNIST/N-CALTECH101 datasets.

    Code adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py
    """
    if is_stream:
        raw_data = np.frombuffer(bin_file.read(), dtype=np.uint8).astype(np.uint32)
    else:
        with open(bin_file, "rb") as fp:
            raw_data = np.fromfile(fp, dtype=np.uint8).astype(np.uint32)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    # Process time stamp overflow events
    time_increment = 2**13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    xytp = make_structured_array(
        all_x[td_indices],
        all_y[td_indices],
        all_ts[td_indices],
        all_p[td_indices],
        dtype=dtype,
    )

    return xytp[:max_n_events]


def read_mnist_file_by_time(
    bin_file: Union[str, BinaryIO], dtype: np.dtype, is_stream: bool = False, t_min=int, t_max = int):

    if is_stream:
        raw_data = np.frombuffer(bin_file.read(), dtype=np.uint8).astype(np.uint32)
    else:
        with open(bin_file, "rb") as fp:
            raw_data = np.fromfile(fp, dtype=np.uint8).astype(np.uint32)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    # Process time stamp overflow events
    time_increment = 2**13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    xytp = make_structured_array(
        all_x[td_indices],
        all_y[td_indices],
        all_ts[td_indices],
        all_p[td_indices],
        dtype=dtype,
    )

    mask = np.logical_and(t_min < xytp['t'], xytp['t'] < t_max)
    x = xytp['x'][mask]
    y = xytp['y'][mask]
    t = xytp['t'][mask]
    p = xytp['p'][mask]

    data = np.zeros((len(x),4))
    data[:,0] = x/240
    data[:,1] = y/240
    data[:,2] = (t-t_min)/(t_max-t_min)
    data[:,3] = p

    return data
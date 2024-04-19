import numpy as np
import matplotlib.pyplot as plt

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
    
    saccades1[:,3] = saccades1[:,3] - 0 #normalize time
    saccades1[:,3] = saccades1[:,3] / 105000   
    #saccades1[:,3] = saccades1[:,3]/100000 #normalize time
    saccades1[:,0] = saccades1[:,0]/33 #normalize coordinate
    saccades1[:,1] = saccades1[:,1]/33
    #saccades1[:,2][saccades1[:,2]==0] = -1 #convert polarity 0 to -1

    #[x,y,t,p]
    saccades2 = np.array(saccades1.tolist(), dtype=float) #transfer dtype
    saccades2[:,2] = saccades1[:,3]
    saccades2[:,3] = saccades1[:,2]
    return saccades2
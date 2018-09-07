import numpy as np
import flowio
from collections import namedtuple
from FlowCal.io import FCSData

FCSFile = namedtuple('FCSFile', ['data', 'channels'])


def read_fcs_data(file_path):

    loaded_fcs = FCSData(file_path)
    loaded_channels = flowio.FlowData(file_path).channels

    channels = list()

    for index in loaded_channels:
        channels.append(loaded_channels[index]['PnS'])

    return FCSFile(data=np.array(loaded_fcs), channels=channels)


def extract_marker_indices(fcs_data, markers_of_interest):

    marker_indices = [fcs_data.channels.index(name) for name in markers_of_interest]

    return marker_indices

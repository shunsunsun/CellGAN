import numpy as np
import flowio
from collections import namedtuple
from FlowCal.io import FCSData

# FCSFile is a named tuple with two attributes, data and channels
# data: Data from the Flow cytometry experiment
# channels: Channels tested in the experiment (here, markers)

FCSFile = namedtuple('FCSFile', ['data', 'channels'])


def read_fcs_data(file_path):

    """
    Reads flow cytometry from given path and stores it in the FCSFile named tuple
    :param file_path: path to given .fcs file
    :return: FCSFile, with data and channels attributes
    """

    if file_path.split('.')[-1] != 'fcs':
        raise NotImplementedError('Please submit a .fcs file for loading')

    try:
        loaded_fcs = FCSData(file_path)
        loaded_channels = flowio.FlowData(file_path).channels
        channels = list()

        for index in loaded_channels:
            channels.append(loaded_channels[index]['PnS'])

        return FCSFile(data=np.array(loaded_fcs), channels=channels)

    except:
        KeyError


def extract_marker_indices(fcs_data, markers_of_interest):

    """
    Extracts indices of markers of interest from given FCSFile namedtuple
    :param fcs_data: FCSFile named_tuple
    :param markers_of_interest: list of markers of interest
    :return: marker_indices, list of indices where corresponding markers are present
    """

    marker_indices = [fcs_data.channels.index(name) for name in markers_of_interest]

    return marker_indices

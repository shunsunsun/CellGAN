import numpy as np
import flowio
from collections import namedtuple
from FlowCal.io import FCSData
import os

# FCSFile is a named tuple with two attributes, data and channels
# data: Data from the Flow cytometry experiment
# channels: Channels tested in the experiment (here, markers)

FCSFile = namedtuple('FCSFile', ['data', 'channels'])


def read_fcs_data(file_path):
    """
    Reads flow cytometry from given path and stores it in the FCSFile named tuple.
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
    except KeyError:
        pass


def extract_marker_indices(fcs_data, markers_of_interest):
    """
    Extracts indices of markers of interest from given FCSFile namedtuple
    :param fcs_data: FCSFile named_tuple
    :param markers_of_interest: list of markers of interest
    :return: marker_indices, list of indices where corresponding markers are present
    """
    marker_indices = [fcs_data.channels.index(name) for name in markers_of_interest]
    return marker_indices


def load_fcs(fcs_files, markers, args, logger=None):
    """Loads the given fcs files and returns the training set. """
    from cellgan.lib.utils import f_trans
    celltype_added = 0
    training_data = list()
    training_labels = list()

    for file in fcs_files:
        file_path = os.path.join(args.input_dir, file.strip())
        fcs_data = read_fcs_data(file_path=file_path)
        try:
            marker_indices = extract_marker_indices(
                fcs_data=fcs_data, markers_of_interest=markers)
            num_cells_in_file = fcs_data.data.shape[0]

            if num_cells_in_file >= args.subpopulation_limit:
                processed_data = np.squeeze(fcs_data.data[:, marker_indices])
                processed_data = f_trans(processed_data, c=args.cofactor)

                training_labels.append([celltype_added] * num_cells_in_file)
                celltype_added += 1
                training_data.append(processed_data)

                if logger is not None:
                    logger.info('File {} loaded and processed'.format(file))
                    logger.info('File {} contains {} cells \n'.format(file, num_cells_in_file))
            else:
                continue
        except AttributeError:
            pass

    return training_data, training_labels

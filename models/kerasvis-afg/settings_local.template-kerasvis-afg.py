# Define critical settings and/or override defaults specified in
# settings.py. Copy this file to settings_local.py in the same
# directory as settings.py and edit. Any settings defined here
# will override those defined in settings.py

import numpy as np

from wave_detection.spindle import DetectSpindle

window_panes = (
    # (i, j, i_size, j_size)
    ('input', (0, 0, 300, 300)),  # This pane is required to show the input picture
    ('input_label', (300, 0, 50, 300)),
    ('kerasvis_aux', (350, 0, 300, 300)),
    ('kerasvis_back', (650, 0, 250, 300)),
    ('kerasvis_status', (900, 0, 30, 1500)),
    ('kerasvis_control', (0, 300, 90, 900)),
    ('kerasvis_layers', (90, 300, 810, 900)),
    ('kerasvis_jpgvis', (0, 1200, 900, 300)),
)

# static_files_dir = '%DVT_ROOT%/models/kerasvis-afg/patients'
# static_files_dir = '%DVT_ROOT%/models/kerasvis-afg/patients_subset'
# static_files_dir = '%DVT_ROOT%/models/kerasvis-afg/patients_s2'
static_files_dir = '%DVT_ROOT%/models/kerasvis-afg/patients_wavelet'

#  Fpz-Cz <- location on the head and EEG is measuring potential in electricity
signal_labels = ['EEG Fpz-Cz',
                 'EEG Pz-Oz',
                 'EOG horizontal']

# Other optional settings; see complete documentation for each in settings.py.
kerasvis_labels = '%DVT_ROOT%/models/kerasvis-afg/deepsleep_labels.txt'

# Load model: kerasvis-afg
# Path to the h5 deploy file.
kerasvis_deploy_model = '%DVT_ROOT%/models/kerasvis-afg/model.h5'
# kerasvis_deploy_model = '%DVT_ROOT%/models/kerasvis-afg/mike-models/encoder.noID.leaky-depth-6.h5'

# Other optional settings; see complete documentation for each in settings.py.
kerasvis_label_layers = ('fc8', 'prob')
kerasvis_prob_layer = 'prob'
kerasvis_unit_jpg_dir = '%DVT_ROOT%/models/caffenet-yos/unit_jpg_vis'
kerasvis_jpgvis_layers = []
kerasvis_jpgvis_remap = {'activation_1': 'conv1', 'activation_3': 'conv3', 'activation_5': 'conv5'}


def signal_filter_fn(data):
    detsp = DetectSpindle(method='Wamsley2012', frequency=(11, 16))
    sp = detsp(data[:, 0:2])

    # print ('Events found', len(sp.events))
    if len(sp.events) == 0:
        return data, {}

    markers = {}
    filtered = np.array(data, copy=True)
    for ev in sp.events:
        start, end, channels = int(ev['start'] * 100), int(ev['end'] * 100), ev['chan']
        for ch in channels.split(','):
            ch = int(ch)
            filtered[start:end, ch] = np.zeros(end - start)
            if ch not in markers:
                markers[ch] = []
            markers[ch].append(start)
            markers[ch].append(end)
    return filtered, markers


def static_files_data_fn(data):
    # x_raw = data['X'][:,6000:9000,0]
    # x_scaled = (x_raw + 199) / 399
    # return x_scaled
    return data['X']


def static_files_labels_fn(data):
    return data['Y'] if 'Y' in data else None


def static_files_extra_fn(data):
    return data['extra'] if 'extra' in data else None


def kerasvis_layer_pretty_name_fn(name):
    return name.replace('conv1d_', 'conv').replace('batch_normalization_', 'bn') \
        .replace('max_pooling1d_', 'pool').replace('activation_', 'act') \
        .replace('dense_', 'fc').replace('flatten_', 'flat')


# Display tweaks.
# Scale all window panes in UI by this factor
global_scale = 1.0
# Scale all fonts by this factor
global_font_size = 0.7

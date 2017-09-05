# Define critical settings and/or override defaults specified in
# settings.py. Copy this file to settings_local.py in the same
# directory as settings.py and edit. Any settings defined here
# will override those defined in settings.py


window_panes = (
    # (i, j, i_size, j_size)
    ('input', (0, 0, 300, 300)),  # This pane is required to show the input picture
    ('kerasvis_aux', (300, 0, 300, 300)),
    ('kerasvis_back', (600, 0, 300, 300)),
    ('kerasvis_status', (900, 0, 30, 1500)),
    ('kerasvis_control', (0, 300, 90, 900)),
    ('kerasvis_layers', (90, 300, 810, 900)),
    ('kerasvis_jpgvis', (0, 1200, 900, 300)),
)

static_files_dir = '%DVT_ROOT%/models/kerasvis-afg/patients'

#  Fpz-Cz <- location on the head and EEG is measuring potential in electricity
signal_labels = ['EEG Fpz-Cz',
                 'EEG Pz-Oz',
                 'EOG horizontal']

# Other optional settings; see complete documentation for each in settings.py.
kerasvis_labels = '%DVT_ROOT%/models/kerasvis-afg/deepsleep_labels.txt'

# Load model: kerasvis-afg
# Path to the h5 deploy file.
kerasvis_deploy_model = '%DVT_ROOT%/models/kerasvis-afg/model.h5'

# Other optional settings; see complete documentation for each in settings.py.
kerasvis_label_layers = ('fc8', 'prob')
kerasvis_prob_layer = 'prob'
kerasvis_unit_jpg_dir = '%DVT_ROOT%/models/caffenet-yos/unit_jpg_vis'
kerasvis_jpgvis_layers = ['conv1', 'conv3', 'conv5', 'activation_9', 'activation_16', 'activation_17', 'activation_18',
                          'dense_3']
kerasvis_jpgvis_remap = {'activation_1': 'conv1', 'activation_3': 'conv3', 'activation_5': 'conv5'}


def static_files_data_fn(data):
    return data['X']


def static_files_labels_fn(data):
    return data['Y']


def kerasvis_layer_pretty_name_fn(name):
    return name.replace('conv1d_', 'conv').replace('batch_normalization_', 'bn')\
        .replace('max_pooling1d_','pool').replace('activation_', 'act')\
        .replace('dense_', 'fc').replace('flatten_', 'flat')


# Display tweaks.
# Scale all window panes in UI by this factor
global_scale = 0.85
# Scale all fonts by this factor
global_font_size = 0.7

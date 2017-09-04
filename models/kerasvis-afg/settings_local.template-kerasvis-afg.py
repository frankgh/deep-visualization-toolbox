# Define critical settings and/or override defaults specified in
# settings.py. Copy this file to settings_local.py in the same
# directory as settings.py and edit. Any settings defined here
# will override those defined in settings.py


# Path to the model
kerasvis_deploy_model = '%DVT_ROOT%/models/kerasvis-afg/model.h5'

# Other optional settings; see complete documentation for each in settings.py.
kerasvis_labels = '%DVT_ROOT%/models/kerasvis-afg/deepsleep_labels.txt'



# Other optional settings; see complete documentation for each in settings.py.
kerasvis_label_layers = ('fc8', 'prob')
kerasvis_prob_layer = 'prob'
kerasvis_unit_jpg_dir = '%DVT_ROOT%/models/caffenet-yos/unit_jpg_vis'
kerasvis_jpgvis_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'prob']
kerasvis_jpgvis_remap = {'pool1': 'conv1', 'pool2': 'conv2', 'pool5': 'conv5'}


def kerasvis_layer_pretty_name_fn(name):
    return name.replace('pool', 'p').replace('norm', 'n')


# Display tweaks.
# Scale all window panes in UI by this factor
global_scale = 0.7
# Scale all fonts by this factor
# global_font_size = 1.0
static_files_dir = '/Users/francisco/src/deep-visualization-toolbox/input_images/'
input_updater_sleep_after_read_frame = 1.0 / 20

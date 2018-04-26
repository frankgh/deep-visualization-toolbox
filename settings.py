# Settings for Deep Visualization Toolbox
#
# Note: Probably don't change anything in this file. To override
# settings, define them in settings_local.py rather than changing them
# here.

import os
import sys

# Import local / overridden settings. Turn off creation of settings_local.pyc to avoid stale settings if settings_local.py is removed.
sys.dont_write_bytecode = True
try:
    from settings_local import *
except ImportError:
    if not os.path.exists('settings_local.py'):
        raise Exception(
            'Could not import settings_local. Did you create it from the template? See README and start with:\n\n'
            '$ cp models/templates/settings_local.template-kerasvis-afg.py settings_local.py')
    else:
        raise
# Resume usual pyc creation
sys.dont_write_bytecode = False

####################################
#
#  General settings
#
####################################

# How long to sleep in the input reading thread after reading a frame from the camera
input_updater_sleep_after_read_frame = locals().get('input_updater_sleep_after_read_frame', 1.0 / 20)

# Input updater thread die after this many seconds without a heartbeat. Useful during debugging to avoid other threads
# running after main thread has crashed.
input_updater_heartbeat_required = locals().get('input_updater_heartbeat_required', 15.0)

# How long to sleep while waiting for key presses and redraws. Recommendation: 1 (min: 1)
main_loop_sleep_ms = locals().get('main_loop_sleep_ms', 1)

# Whether or not to print a "." every second time through the main loop to visualize the loop rate
print_dots = locals().get('print_dots', False)

####################################
#
#  Window pane layout and colors/fonts
#
####################################

# Show border for each panel and annotate each with its name. Useful
# for debugging window_panes arrangement.
debug_window_panes = locals().get('debug_window_panes', False)

# The window panes available and their layout is determined by the
# "window_panes" variable. By default all panes are enabled with a
# standard size. This setting will often be overridden on a per-model
# basis, e.g. if the model does not have pre-computed jpgvis
# information, the kerasvis_jpgvis pane can be omitted. For
# convenience, if the only variable that needs to be overridden is the
# height of the control panel (to accomodate varying length of layer
# names), one can simply define control_pane_height. If more
if 'default_window_panes' in locals():
    raise Exception('Override window panes in settings_local.py by defining window_panes, not default_window_panes')
default_window_panes = (
    # (i, j, i_size, j_size)
    ('input', (0, 0, 300, 300)),  # This pane is required to show the input picture
    ('kerasvis_aux', (300, 0, 300, 300)),
    ('kerasvis_back', (600, 0, 300, 300)),
    ('kerasvis_status', (900, 0, 30, 1500)),
    ('kerasvis_control', (0, 300, 30, 900)),
    ('kerasvis_layers', (30, 300, 870, 900)),
    ('kerasvis_jpgvis', (0, 1200, 900, 300)),
)
window_panes = locals().get('window_panes', default_window_panes)

# Define global_scale as a float to rescale window and all
# panes. Handy for quickly changing resolution for a different screen.
global_scale = locals().get('global_scale', 1.0)

# Define global_font_size to scale all font sizes by this amount.
global_font_size = locals().get('global_font_size', 1.0)

if global_scale != 1.0:
    scaled_window_panes = []
    for wp in window_panes:
        scaled_window_panes.append([wp[0], [int(val * global_scale) for val in wp[1]]])
    window_panes = scaled_window_panes

# All window configuration information is now contained in the
# window_panes variable. Print if desired:
if debug_window_panes:
    print 'Final window panes and locations/sizes (i, j, i_size, j_size):'
    for pane in window_panes:
        print '  Pane: %s' % repr(pane)

help_pane_loc = locals().get('help_pane_loc', (.07, .07, .86, .86))  # as a fraction of main window
window_background = locals().get('window_background', (.2, .2, .2))
stale_background = locals().get('stale_background', (.3, .3, .2))
static_files_dir = locals().get('static_files_dir', 'input_signals')
static_files_regexp = locals().get('static_files_regexp', '.*\.(npz|npy)$')
static_files_ignore_case = locals().get('static_files_ignore_case', True)
# True to stretch to square, False to crop to square. (Can change at
# runtime via 'stretch_mode' key.)
static_file_stretch_mode = locals().get('static_file_stretch_mode', True)

# int, 0+. How many times to go through the main loop after a keypress
# before resuming handling frames (0 to handle every frame as it
# arrives). Setting this to a value > 0 can enable more responsive
# keyboard input even when other settings are tuned to maximize the
# framerate. Default: 2
keypress_pause_handle_iterations = locals().get('keypress_pause_handle_iterations', 2)

# int, 0+. How many times to go through the main loop after a keypress
# before resuming redraws (0 to redraw every time it is
# needed). Setting this to a value > 0 can enable more responsive
# keyboard input even when other settings are tuned to maximize the
# framerate. Default: 1
keypress_pause_redraw_iterations = locals().get('keypress_pause_redraw_iterations', 1)

# int, 1+. Force a redraw even when keys are pressed if there have
# been this many passes through the main loop without a redraw due to
# the keypress_pause_redraw_iterations setting combined with many key
# presses. Default: 3.
redraw_at_least_every = locals().get('redraw_at_least_every', 3)

# Tuple of tuples describing the file to import and class from it to
# instantiate for each app to be run. Apps are run and given keys to
# handle in the order specified.
default_installed_apps = (
    ('kerasvis.app', 'KerasVisApp'),
)
installed_apps = locals().get('installed_apps', default_installed_apps)

# Font settings for the help pane. Text is rendered using OpenCV; see
# http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#puttext
# for information on parameters.
help_face = locals().get('help_face', 'FONT_HERSHEY_COMPLEX_SMALL')
help_loc = locals().get('help_loc', (20, 10))  # r,c order
help_line_spacing = locals().get('help_line_spacing', 10)  # extra pixel spacing between lines
help_clr = locals().get('help_clr', (1, 1, 1))
help_fsize = locals().get('help_fsize', 1.0 * global_font_size)
help_thick = locals().get('help_thick', 1)

####################################
#
#  Kerasvis settings
#
####################################

# Path to file listing labels in order, one per line, used for the
# below two features. None to disable.
kerasvis_labels = locals().get('kerasvis_labels', None)

# Size of jpg reading cache in bytes (default: 2GB)
# Note: largest fc6/fc7 images are ~600MB. Cache smaller than this will be painfully slow when using patterns_mode for fc6 and fc7.
# Cache use when all layers have been loaded is ~1.6GB
kerasvis_jpg_cache_size = locals().get('kerasvis_jpg_cache_size', 2000 * 1024 ** 2)

####################################
#
#  Kerasvis settings
#
####################################

# Which layers have channels/neurons corresponding to the order given
# in the kerasvis_labels file? Annotate these units with label text
# (when those neurons are selected). None to disable.
kerasvis_label_layers = locals().get('kerasvis_label_layers', None)

# Which layer to use for displaying class output numbers in left pane
# (when no neurons are selected). None to disable.
kerasvis_prob_layer = locals().get('kerasvis_prob_layer', None)

# String or None. Which directory to load pre-computed per-unit
# visualizations from, if any. None to disable.
kerasvis_unit_jpg_dir = locals().get('kerasvis_unit_jpg_dir', None)

# List. For which layers should jpgs be loaded for
# visualization? If a layer name (full name, not prettified) is given
# here, we will try to load jpgs to visualize each unit. This is used
# for pattern mode ('s' key by default) and for the right
# kerasvis_jpgvis pane ('9' key by default). Empty list to disable.
kerasvis_jpgvis_layers = locals().get('kerasvis_jpgvis_layers', [])

# Dict specifying string:string mapping. Steal pattern mode and right
# jpgvis pane visualizations for certain layers (e.g. pool1) from
# other layers (e.g. conv1). We can do this because
# optimization/max-act/deconv-of-max results are identical.
kerasvis_jpgvis_remap = locals().get('kerasvis_jpgvis_remap', {})

# Function mapping old name -> new name to modify/prettify/shorten
# layer names.
kerasvis_layer_pretty_name_fn = locals().get('kerasvis_layer_pretty_name_fn', lambda name: name)

# The KerasVisApp computes a layout of neurons for the kerasvis_layers
# pane given the aspect ratio in kerasvis_layers_aspect_ratio (< 1 for
# portrait, 1 for square, > 1 for landscape). Default: 1 (square).
kerasvis_layers_aspect_ratio = locals().get('kerasvis_layers_aspect_ratio', 1.0)

# Replace magic '%DVT_ROOT%' string with the root DeepVis Toolbox
# directory (the location of this settings file)
dvt_root = os.path.dirname(os.path.abspath(__file__))

if 'kerasvis_deploy_model' in locals():
    kerasvis_deploy_model = kerasvis_deploy_model.replace('%DVT_ROOT%', dvt_root)

if isinstance(kerasvis_labels, basestring):
    kerasvis_labels = kerasvis_labels.replace('%DVT_ROOT%', dvt_root)
if isinstance(kerasvis_unit_jpg_dir, basestring):
    kerasvis_unit_jpg_dir = kerasvis_unit_jpg_dir.replace('%DVT_ROOT%', dvt_root)
if isinstance(static_files_dir, basestring):
    static_files_dir = static_files_dir.replace('%DVT_ROOT%', dvt_root)

# Pause Keras forward/backward computation for this many seconds after a keypress. This is to keep the processor free
# for a brief period after a keypress, which allow the interface to feel much more responsive. After this period has
# passed, Keras resumes computation, in CPU mode often occupying all cores. Default: .1
kerasvis_pause_after_keys = locals().get('kerasvis_pause_after_keys', .10)
kerasvis_frame_wait_sleep = locals().get('kerasvis_frame_wait_sleep', .01)
kerasvis_jpg_load_sleep = locals().get('kerasvis_jpg_load_sleep', .01)
# KerasProc thread dies after this many seconds without a
# heartbeat. Useful during debugging to avoid other threads running
# after main thread has crashed.
kerasvis_heartbeat_required = locals().get('kerasvis_heartbeat_required', 15.0)

# How far to move when using fast left/right/up/down keys
kerasvis_fast_move_dist = locals().get('kerasvis_fast_move_dist', 3)

kerasvis_grad_norm_blur_radius = locals().get('kerasvis_grad_norm_blur_radius', 4.0)

# Boost display of individual channels. For channel activations in the
# range [0,1], boost_indiv rescales the activations of that channel
# such that the new_max = old_max ** -boost_indiv. Thus no-op value =
# 0.0, and a value of 1.0 means each channel is scaled to use the
# entire [0,1] range.
kerasvis_boost_indiv_choices = locals().get('kerasvis_boost_indiv_choices', (0, .3, .5, .8, 1))
# Default boost indiv given as index into kerasvis_boost_indiv_choices
kerasvis_boost_indiv_default_idx = locals().get('kerasvis_boost_indiv_default_idx', 0)
# Boost display of entire layer activation by the given gamma value
# (for values in [0,1], display_val = old_val ** gamma. No-op value:
# 1.0)
kerasvis_boost_gamma_choices = locals().get('kerasvis_boost_gamma_choices', (1, .7, .5, .3))
# Default boost gamma given as index into kerasvis_boost_gamma_choices
kerasvis_boost_gamma_default_idx = locals().get('kerasvis_boost_gamma_default_idx', 0)
# Initially show label predictions or not (toggle with default key '8')
kerasvis_init_show_label_predictions = locals().get('kerasvis_init_show_label_predictions', True)
# Initially show jpg vis or not (toggle with default key '9')
kerasvis_init_show_unit_jpgs = locals().get('kerasvis_init_show_unit_jpgs', True)

# extra pixel spacing between lines. Default: 4 = not much space / tight layout
kerasvis_control_line_spacing = locals().get('kerasvis_control_line_spacing', 4)
# Font settings for control pane (list of layers)
kerasvis_control_face = locals().get('kerasvis_control_face', 'FONT_HERSHEY_COMPLEX_SMALL')
kerasvis_control_loc = locals().get('kerasvis_control_loc', (15, 5))  # r,c order
kerasvis_control_clr = locals().get('kerasvis_control_clr', (.8, .8, .8))
kerasvis_control_clr_selected = locals().get('kerasvis_control_clr_selected', (1, 1, 1))
kerasvis_control_clr_cursor = locals().get('kerasvis_control_clr_cursor', (.5, 1, .5))
kerasvis_control_clr_bp = locals().get('kerasvis_control_clr_bp', (.8, .8, 1))
kerasvis_control_fsize = locals().get('kerasvis_control_fsize', 1.0 * global_font_size)
kerasvis_control_thick = locals().get('kerasvis_control_thick', 1)
kerasvis_control_thick_selected = locals().get('kerasvis_control_thick_selected', 2)
kerasvis_control_thick_cursor = locals().get('kerasvis_control_thick_cursor', 2)
kerasvis_control_thick_bp = locals().get('kerasvis_control_thick_bp', 2)

# Color settings for layer activation pane
kerasvis_layer_clr_cursor = locals().get('kerasvis_layer_clr_cursor', (.5, 1, .5))
kerasvis_layer_clr_back_background = locals().get('kerasvis_layer_clr_back_background', (.2, .2, .5))
kerasvis_layer_clr_back_sel = locals().get('kerasvis_layer_clr_back_sel', (.2, .2, 1))

# Font settings for status pane (bottom line)
kerasvis_status_face = locals().get('kerasvis_status_face', 'FONT_HERSHEY_COMPLEX_SMALL')
kerasvis_status_loc = locals().get('kerasvis_status_loc', (15, 10))  # r,c order
kerasvis_status_line_spacing = locals().get('kerasvis_status_line_spacing', 5)  # extra pixel spacing between lines
kerasvis_status_clr = locals().get('kerasvis_status_clr', (.8, .8, .8))
kerasvis_status_fsize = locals().get('kerasvis_status_fsize', 1.0 * global_font_size)
kerasvis_status_thick = locals().get('kerasvis_status_thick', 1)
kerasvis_jpgvis_stack_vert = locals().get('kerasvis_jpgvis_stack_vert', True)

# Font settings for class prob output (top 5 classes listed on left)
kerasvis_class_face = locals().get('kerasvis_class_face', 'FONT_HERSHEY_COMPLEX_SMALL')
kerasvis_class_loc = locals().get('kerasvis_class_loc', (20, 10))  # r,c order
kerasvis_class_line_spacing = locals().get('kerasvis_class_line_spacing', 10)  # extra pixel spacing between lines
kerasvis_class_clr_0 = locals().get('kerasvis_class_clr_0', (.5, .5, .5))
kerasvis_class_clr_1 = locals().get('kerasvis_class_clr_1', (.5, 1, .5))
kerasvis_class_fsize = locals().get('kerasvis_class_fsize', 1.0 * global_font_size)
kerasvis_class_thick = locals().get('kerasvis_class_thick', 1)

# Font settings for label overlay text (shown on layer pane only for kerasvis_label_layers layers)
kerasvis_label_face = locals().get('kerasvis_label_face', 'FONT_HERSHEY_COMPLEX_SMALL')
kerasvis_label_loc = locals().get('kerasvis_label_loc', (30, 20))  # r,c order
kerasvis_label_clr = locals().get('kerasvis_label_clr', (.8, .8, .8))
kerasvis_label_fsize = locals().get('kerasvis_label_fsize', 1.0 * global_font_size)
kerasvis_label_thick = locals().get('kerasvis_label_thick', 1)

####################################
#
#  A few final sanity checks
#
####################################

# Check that required setting have been defined
bound_locals = locals()


def assert_in_settings(setting_name):
    if not setting_name in bound_locals:
        raise Exception('The "%s" setting is required; be sure to define it in settings_local.py' % setting_name)


assert_in_settings('kerasvis_deploy_model')

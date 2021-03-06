#! /usr/bin/env python
# -*- coding: utf-8

import StringIO
import os

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model

from app_base import BaseApp
from image_misc import FormattedString, cv2_typeset_text, to_255, norm01, norm01c, tile_images_normalize, \
    ensure_float01, tile_images_make_tiles, ensure_uint255_and_resize_to_fit, get_tiles_height_width_ratio, \
    plt_plot_filters_blit, plt_plot_filter, plt_plot_heatmap
from jpg_vis_loading_thread import JPGVisLoadingThread
from kerasvis.keras_proc_thread import KerasProcThread
from kerasvis_app_state import KerasVisAppState
from kerasvis_helper import get_pretty_layer_name, read_label_file, load_square_sprite_image
from misc import WithTimer
from numpy_cache import FIFOLimitedArrayCache


class KerasVisApp(BaseApp):
    '''App to visualize using keras.'''

    def __init__(self, settings, key_bindings):
        super(KerasVisApp, self).__init__(settings, key_bindings)
        print 'Got settings', settings
        self.settings = settings
        self.bindings = key_bindings

        self._net_channel_swap = (2, 1, 0)
        self._net_channel_swap_inv = tuple(
            [self._net_channel_swap.index(ii) for ii in range(len(self._net_channel_swap))])
        self._range_scale = 1.0  # not needed; image already in [0,255]

        if isinstance(settings.kerasvis_deploy_model, basestring):
            self.net = load_model(settings.kerasvis_deploy_model)
            # self.net._make_predict_function() This doesn't look necessary
            self.graph = tf.get_default_graph()
            print ('KerasVisApp: model has been loaded')
        elif isinstance(settings.kerasvis_deploy_model, list):
            self.nets = [load_model(m) for m in settings.kerasvis_deploy_model]

        self.labels = None
        if self.settings.kerasvis_labels:
            self.labels = read_label_file(self.settings.kerasvis_labels)
        self.proc_thread = None
        self.jpgvis_thread = None
        self.handled_frames = 0
        if settings.kerasvis_jpg_cache_size < 10 * 1024 ** 2:
            raise Exception('kerasvis_jpg_cache_size must be at least 10MB for normal operation.')
        self.img_cache = FIFOLimitedArrayCache(settings.kerasvis_jpg_cache_size)

        self._populate_net_layer_info()

    def _populate_net_layer_info(self):
        '''For each layer, save the number of filters and precompute
        tile arrangement (needed by KerasVisAppState to handle
        keyboard navigation).
        '''
        self.net_layer_info = {}

        for layer in self.net.layers:
            self.net_layer_info[layer.name] = {}
            blob_shape = layer.output_shape
            self.net_layer_info[layer.name]['isconv'] = isinstance(layer, keras.layers.convolutional._Conv)
            self.net_layer_info[layer.name]['data_shape'] = blob_shape[1:]  # Chop off batch size
            self.net_layer_info[layer.name]['n_tiles'] = blob_shape[-1]
            self.net_layer_info[layer.name]['tiles_rc'] = get_tiles_height_width_ratio(
                self.net_layer_info[layer.name]['n_tiles'], self.settings.kerasvis_layers_aspect_ratio)
            self.net_layer_info[layer.name]['tile_rows'] = self.net_layer_info[layer.name]['tiles_rc'][0]
            self.net_layer_info[layer.name]['tile_cols'] = self.net_layer_info[layer.name]['tiles_rc'][1]

    def start(self):
        self.state = KerasVisAppState(self.net, self.settings, self.bindings, self.net_layer_info)
        self.state.drawing_stale = True
        self.layer_print_names = [get_pretty_layer_name(self.settings, nn) for nn in self.state._layers]

        if self.proc_thread is None or not self.proc_thread.is_alive():
            # Start thread if it's not already running
            self.proc_thread = KerasProcThread(self.net, self.graph, self.state,
                                               self.settings.kerasvis_frame_wait_sleep,
                                               self.settings.kerasvis_pause_after_keys,
                                               self.settings.kerasvis_heartbeat_required)
            self.proc_thread.start()

        if self.jpgvis_thread is None or not self.jpgvis_thread.is_alive():
            # Start thread if it's not already running
            self.jpgvis_thread = JPGVisLoadingThread(self.settings, self.state, self.img_cache,
                                                     self.settings.kerasvis_jpg_load_sleep,
                                                     self.settings.kerasvis_heartbeat_required)
            self.jpgvis_thread.start()

    def get_heartbeats(self):
        return [self.proc_thread.heartbeat, self.jpgvis_thread.heartbeat]

    def quit(self):
        print ('KerasVisApp: trying to quit')

        with self.state.lock:
            self.state.quit = True

        if self.proc_thread != None:
            for ii in range(3):
                self.proc_thread.join(1)
                if not self.proc_thread.is_alive():
                    break
            if self.proc_thread.is_alive():
                raise Exception('KerasVisApp: Could not join proc_thread; giving up.')
            self.proc_thread = None

        print 'KerasVisApp: quitting.'

    def _can_skip_all(self, panes):
        return ('kerasvis_layers' not in panes.keys())

    def handle_input(self, input_signal, extra_info, panes):
        if self.debug_level > 1:
            print 'handle_input: signal number {} is {}'.format(self.handled_frames,
                                                                'None' if input_signal is None else 'Available')
        self.handled_frames += 1
        if self._can_skip_all(panes):
            return

        with self.state.lock:
            if self.debug_level > 1:
                print ('KerasVisApp.handle_input: pushed frame')
            self.state.next_frame = input_signal
            self.state.active_signal = input_signal
            self.state.extra_info = extra_info
            if self.debug_level > 1:
                print ('KerasVisApp.handle_input: keras_net_state is: {}'.format(self.state.keras_net_state))

    def redraw_needed(self):
        return self.state.redraw_needed()

    def draw(self, panes):
        if self._can_skip_all(panes):
            if self.debug_level > 1:
                print 'KerasVisApp.draw: skipping'
            return False

        with self.state.lock:
            # Hold lock throughout drawing
            do_draw = self.state.drawing_stale and self.state.keras_net_state == 'free'
            if self.debug_level == 3:
                print 'KerasProcThread.draw: keras_net_state is: {}'.format(self.state.keras_net_state)
            if do_draw:
                self.state.keras_net_state = 'draw'

        if do_draw:
            if self.debug_level > 1:
                print 'KerasVisApp.draw: drawing'

            if 'kerasvis_control' in panes:
                self._draw_control_pane(panes['kerasvis_control'])
            layer_data_3D_highres, selected_unit_highres = None, None
            if 'kerasvis_layers' in panes:
                layer_data_3D_highres, selected_unit_highres = self._draw_layer_pane(panes['kerasvis_layers'])
            if 'kerasvis_selected' in panes:
                self._draw_selected_pane(panes['kerasvis_selected'], layer_data_3D_highres, selected_unit_highres)
                self._draw_aux_pane(panes['kerasvis_aux'], None)
            elif 'kerasvis_aux' in panes:
                self._draw_aux_pane(panes['kerasvis_aux'], layer_data_3D_highres, selected_unit_highres)
            if 'kerasvis_back' in panes:
                # Draw back pane as normal
                self._draw_back_pane(panes['kerasvis_back'])
                if self.state.layers_pane_zoom_mode == 2:
                    # ALSO draw back pane into layers pane
                    self._draw_back_pane(panes['kerasvis_layers'])
            if 'kerasvis_jpgvis' in panes:
                self._draw_jpgvis_pane(panes['kerasvis_jpgvis'])
            if 'kerasvis_status' in panes:
                self._draw_status_pane(panes['kerasvis_status'])

            with self.state.lock:
                self.state.drawing_stale = False
                self.state.keras_net_state = 'free'
        return do_draw

    def _draw_prob_labels_pane(self, pane):
        '''Adds text label annotation atop the given pane.'''

        if not self.labels or not self.state.show_label_predictions or not self.settings.kerasvis_prob_layer:
            return

        # pane.data[:] = to_255(self.settings.window_background)
        defaults = {'face': getattr(cv2, self.settings.kerasvis_class_face),
                    'fsize': self.settings.kerasvis_class_fsize,
                    'clr': to_255(self.settings.kerasvis_class_clr_0),
                    'thick': self.settings.kerasvis_class_thick}
        loc = self.settings.kerasvis_class_loc[::-1]  # Reverse to OpenCV c,r order
        clr_0 = to_255(self.settings.kerasvis_class_clr_0)
        clr_1 = to_255(self.settings.kerasvis_class_clr_1)

        if not hasattr(self.net, 'intermediate_predictions') or \
                self.net.intermediate_predictions is None:
            probs_flat = np.array([0, 0, 0, 0, 0])
        else:
            probs_flat = self.net.intermediate_predictions[-1][0]

        top_5 = probs_flat.argsort()[-1:-6:-1]

        strings = []
        fs = FormattedString('Predicted Label:', defaults)
        fs.clr = clr_0
        strings.append([fs])
        for idx in top_5:
            prob = probs_flat[idx]
            text = '  %.2f %s' % (prob, self.labels[idx])
            fs = FormattedString(text, defaults)
            # fs.clr = tuple([clr_1[ii]*prob/pmax + clr_0[ii]*(1-prob/pmax) for ii in range(3)])
            fs.clr = tuple([max(0, min(255, clr_1[ii] * prob + clr_0[ii] * (1 - prob))) for ii in range(3)])
            strings.append([fs])  # Line contains just fs

        cv2_typeset_text(pane.data, strings, loc, line_spacing=self.settings.kerasvis_class_line_spacing)

    def _draw_control_pane(self, pane):
        pane.data[:] = to_255(self.settings.window_background)

        loc = self.settings.kerasvis_control_loc[::-1]  # Reverse to OpenCV c,r order

        strings = []
        defaults = {'face': getattr(cv2, self.settings.kerasvis_control_face),
                    'fsize': self.settings.kerasvis_control_fsize,
                    'clr': to_255(self.settings.kerasvis_control_clr),
                    'thick': self.settings.kerasvis_control_thick}

        for ii in range(len(self.layer_print_names)):
            fs = FormattedString(self.layer_print_names[ii], defaults)
            this_layer = self.state._layers[ii]
            if self.state.backprop_selection_frozen and this_layer == self.state.backprop_layer:
                fs.clr = to_255(self.settings.kerasvis_control_clr_bp)
                fs.thick = self.settings.kerasvis_control_thick_bp
            if this_layer == self.state.layer:
                if self.state.cursor_area == 'top':
                    fs.clr = to_255(self.settings.kerasvis_control_clr_cursor)
                    fs.thick = self.settings.kerasvis_control_thick_cursor
                else:
                    if not (self.state.backprop_selection_frozen and this_layer == self.state.backprop_layer):
                        fs.clr = to_255(self.settings.kerasvis_control_clr_selected)
                        fs.thick = self.settings.kerasvis_control_thick_selected
            strings.append(fs)

        cv2_typeset_text(pane.data, strings, loc,
                         line_spacing=self.settings.kerasvis_control_line_spacing,
                         wrap=True)

    def _draw_status_pane(self, pane):
        pane.data[:] = to_255(self.settings.window_background)

        defaults = {'face': getattr(cv2, self.settings.kerasvis_status_face),
                    'fsize': self.settings.kerasvis_status_fsize,
                    'clr': to_255(self.settings.kerasvis_status_clr),
                    'thick': self.settings.kerasvis_status_thick}
        loc = self.settings.kerasvis_status_loc[::-1]  # Reverse to OpenCV c,r order

        status = StringIO.StringIO()
        fps = self.proc_thread.approx_fps()
        with self.state.lock:
            print >> status, 'pattern' if self.state.pattern_mode else (
                'back' if self.state.layers_show_back else 'fwd'),
            print >> status, '%s:%d |' % (self.state.layer, self.state.selected_unit),

            filter_mode = self.state.layers_pane_filter_mode

            if filter_mode == 1:
                print >> status, 'View: Average |',
            elif filter_mode == 2:
                print >> status, 'View: Max |',
            elif filter_mode == 3:
                print >> status, 'View: Plot |',
            elif filter_mode == 4:
                print >> status, 'View: Extra |',
            elif filter_mode == 5:
                print >> status, 'View: Heatmap |',
            else:
                print >> status, 'View: Activation |',

            if not self.state.back_enabled:
                print >> status, 'Back: off',
            else:
                print >> status, 'Back: %s' % ('deconv' if self.state.back_mode == 'deconv' else 'bprop'),
                print >> status, '(from %s_%d, disp %s)' % (self.state.backprop_layer,
                                                            self.state.backprop_unit,
                                                            self.state.back_filt_mode),
            print >> status, '|',
            print >> status, 'Boost: %g/%g' % (self.state.layer_boost_indiv, self.state.layer_boost_gamma)

            if fps > 0:
                print >> status, '| FPS: %.01f' % fps

            if self.state.extra_msg:
                print >> status, '|', self.state.extra_msg
                self.state.extra_msg = ''

        strings = [FormattedString(line, defaults) for line in status.getvalue().split('\n')]

        cv2_typeset_text(pane.data, strings, loc,
                         line_spacing=self.settings.kerasvis_status_line_spacing)

    def _draw_layer_pane(self, pane):
        '''Returns the data shown in highres format, b01c order.'''

        if not hasattr(self.net, 'intermediate_predictions') or \
                self.net.intermediate_predictions is None:
            return None, None

        display_3D_highres, selected_unit_highres = None, None
        out = self.net.intermediate_predictions[self.state.layer_idx]

        if self.state.layers_pane_filter_mode in (4, 5) and self.state.extra_info is None:
            self.state.layers_pane_filter_mode = 0

        state_layers_pane_filter_mode = self.state.layers_pane_filter_mode
        assert state_layers_pane_filter_mode in (0, 1, 2, 3, 4)

        # Display pane based on layers_pane_zoom_mode
        state_layers_pane_zoom_mode = self.state.layers_pane_zoom_mode
        assert state_layers_pane_zoom_mode in (0, 1, 2)

        layer_dat_3D = out[0].T
        n_tiles = layer_dat_3D.shape[0]
        tile_rows, tile_cols = self.net_layer_info[self.state.layer]['tiles_rc']

        if state_layers_pane_filter_mode == 0:
            if len(layer_dat_3D.shape) > 1:
                img_width, img_height = get_tiles_height_width_ratio(layer_dat_3D.shape[1],
                                                                     self.settings.kerasvis_layers_aspect_ratio)

                pad = np.zeros((layer_dat_3D.shape[0], ((img_width * img_height) - layer_dat_3D.shape[1])))
                layer_dat_3D = np.concatenate((layer_dat_3D, pad), axis=1)
                layer_dat_3D = np.reshape(layer_dat_3D, (layer_dat_3D.shape[0], img_width, img_height))

        elif state_layers_pane_filter_mode == 1:
            if len(layer_dat_3D.shape) > 1:
                layer_dat_3D = np.average(layer_dat_3D, axis=1)

        elif state_layers_pane_filter_mode == 2:
            if len(layer_dat_3D.shape) > 1:
                layer_dat_3D = np.max(layer_dat_3D, axis=1)

        elif state_layers_pane_filter_mode == 3:

            if len(layer_dat_3D.shape) > 1:
                title, r, c, hide_axis = None, tile_rows, tile_cols, True
                x_axis_label, y_axis_label = None, None
                if self.state.cursor_area == 'bottom' and state_layers_pane_zoom_mode == 1:
                    r, c, hide_axis = 1, 1, False
                    layer_dat_3D = layer_dat_3D[self.state.selected_unit:self.state.selected_unit + 1]
                    title = 'Layer {}, Filter {}'.format(self.state._layers[self.state.layer_idx],
                                                         self.state.selected_unit)
                    x_axis_label, y_axis_label = 'Time', 'Activation'

                display_3D = plt_plot_filters_blit(
                    y=layer_dat_3D,
                    x=None,
                    shape=(pane.data.shape[0], pane.data.shape[1]),
                    rows=r,
                    cols=c,
                    title=title,
                    log_scale=self.state.log_scale,
                    hide_axis=hide_axis,
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label
                )

                if self.state.cursor_area == 'bottom' and state_layers_pane_zoom_mode == 0:
                    selected_unit_highres = plt_plot_filter(
                        x=None,
                        y=layer_dat_3D[self.state.selected_unit],
                        title='Layer {}, Filter {}'.format(self.state._layers[self.state.layer_idx],
                                                           self.state.selected_unit),
                        log_scale=self.state.log_scale,
                        x_axis_label='Time',
                        y_axis_label='Activation'
                    )

            else:
                state_layers_pane_filter_mode = 0

        elif state_layers_pane_filter_mode == 4:

            if self.state.extra_info is not None:
                extra = self.state.extra_info.item()
                is_heatmap = True if 'type' in extra and extra['type'] == 'heatmap' else False

                if is_heatmap:
                    layer_dat_3D = extra['data'][self.state.layer_idx]

                    if self.state.cursor_area == 'bottom' and state_layers_pane_zoom_mode == 1:
                        display_3D = plt_plot_heatmap(
                            data=layer_dat_3D[self.state.selected_unit:self.state.selected_unit + 1],
                            shape=(pane.data.shape[0], pane.data.shape[1]),
                            rows=1,
                            cols=1,
                            x_axis_label=extra['x_axis'],
                            y_axis_label=extra['y_axis'],
                            title='Layer {}, Filter {} \n {}'.format(self.state._layers[self.state.layer_idx],
                                                                     self.state.selected_unit, extra['title']),
                            hide_axis=False,
                            x_axis_values=extra['x_axis_values'],
                            y_axis_values=extra['y_axis_values'],
                            vmin=layer_dat_3D.min(),
                            vmax=layer_dat_3D.max()
                        )
                    else:
                        display_3D = plt_plot_heatmap(
                            data=layer_dat_3D,
                            shape=(pane.data.shape[0], pane.data.shape[1]),
                            rows=tile_rows,
                            cols=tile_cols,
                            x_axis_label=extra['x_axis'],
                            y_axis_label=extra['y_axis'],
                            title=extra['title'],
                            x_axis_values=extra['x_axis_values'],
                            y_axis_values=extra['y_axis_values']
                        )

                    if self.state.cursor_area == 'bottom':
                        selected_unit_highres = plt_plot_heatmap(
                            data=layer_dat_3D[self.state.selected_unit:self.state.selected_unit + 1],
                            shape=(300, 300),
                            rows=1,
                            cols=1,
                            x_axis_label=extra['x_axis'],
                            y_axis_label=extra['y_axis'],
                            title='Layer {}, Filter {} \n {}'.format(self.state._layers[self.state.layer_idx],
                                                                     self.state.selected_unit, extra['title']),
                            x_axis_values=extra['x_axis_values'],
                            y_axis_values=extra['y_axis_values'],
                            hide_axis=False,
                            vmin=layer_dat_3D.min(),
                            vmax=layer_dat_3D.max()
                        )[0]

                else:

                    layer_dat_3D = extra['x'][self.state.layer_idx]
                    title, x_axis_label, y_axis_label, r, c, hide_axis = None, None, None, tile_rows, tile_cols, True

                    if self.state.cursor_area == 'bottom':
                        if state_layers_pane_zoom_mode == 1:
                            r, c, hide_axis = 1, 1, False
                            layer_dat_3D = layer_dat_3D[self.state.selected_unit:self.state.selected_unit + 1]
                            title = 'Layer {}, Filter {} \n {}'.format(self.state._layers[self.state.layer_idx],
                                                                       self.state.selected_unit, extra['title'])
                            x_axis_label, y_axis_label = extra['x_axis'], extra['y_axis']

                            if self.state.log_scale == 1:
                                y_axis_label = y_axis_label + ' (log-scale)'

                    # start_time = timeit.default_timer()
                    display_3D = plt_plot_filters_blit(
                        y=layer_dat_3D,
                        x=extra['y'],
                        shape=(pane.data.shape[0], pane.data.shape[1]),
                        rows=r,
                        cols=c,
                        title=title,
                        log_scale=self.state.log_scale,
                        x_axis_label=x_axis_label,
                        y_axis_label=y_axis_label,
                        hide_axis=hide_axis
                    )

                    if self.state.cursor_area == 'bottom' and state_layers_pane_zoom_mode == 0:
                        selected_unit_highres = plt_plot_filter(
                            x=extra['y'],
                            y=layer_dat_3D[self.state.selected_unit],
                            title='Layer {}, Filter {} \n {}'.format(self.state._layers[self.state.layer_idx],
                                                                     self.state.selected_unit, extra['title']),
                            log_scale=self.state.log_scale,
                            x_axis_label=extra['x_axis'],
                            y_axis_label=extra['y_axis']
                        )

            # TODO

            # if hasattr(self.settings, 'static_files_extra_fn'):
            #     self.data = self.settings.static_files_extra_fn(self.latest_static_file)
            #      self.state.layer_idx

        if len(layer_dat_3D.shape) == 1:
            layer_dat_3D = layer_dat_3D[:, np.newaxis, np.newaxis]

        if self.state.layers_show_back and not self.state.pattern_mode:
            padval = self.settings.kerasvis_layer_clr_back_background
        else:
            padval = self.settings.window_background

        if self.state.pattern_mode:
            # Show desired patterns loaded from disk

            load_layer = self.state.layer
            if self.settings.kerasvis_jpgvis_remap and self.state.layer in self.settings.kerasvis_jpgvis_remap:
                load_layer = self.settings.kerasvis_jpgvis_remap[self.state.layer]

            if self.settings.kerasvis_jpgvis_layers and load_layer in self.settings.kerasvis_jpgvis_layers:
                jpg_path = os.path.join(self.settings.kerasvis_unit_jpg_dir,
                                        'regularized_opt', load_layer, 'whole_layer.jpg')

                # Get highres version
                # cache_before = str(self.img_cache)
                display_3D_highres = self.img_cache.get((jpg_path, 'whole'), None)
                # else:
                #    display_3D_highres = None

                if display_3D_highres is None:
                    try:
                        with WithTimer('KerasVisApp:load_sprite_image', quiet=self.debug_level < 1):
                            display_3D_highres = load_square_sprite_image(jpg_path, n_sprites=n_tiles)
                    except IOError:
                        # File does not exist, so just display disabled.
                        pass
                    else:
                        self.img_cache.set((jpg_path, 'whole'), display_3D_highres)
                        # cache_after = str(self.img_cache)
                        # print 'Cache was / is:\n  %s\n  %s' % (cache_before, cache_after)

            if display_3D_highres is not None:
                # Get lowres version, maybe. Assume we want at least one pixel for selection border.
                row_downsamp_factor = int(
                    np.ceil(float(display_3D_highres.shape[1]) / (pane.data.shape[0] / tile_rows - 2)))
                col_downsamp_factor = int(
                    np.ceil(float(display_3D_highres.shape[2]) / (pane.data.shape[1] / tile_cols - 2)))
                ds = max(row_downsamp_factor, col_downsamp_factor)
                if ds > 1:
                    # print 'Downsampling by', ds
                    display_3D = display_3D_highres[:, ::ds, ::ds, :]
                else:
                    display_3D = display_3D_highres
            else:
                display_3D = layer_dat_3D * 0  # nothing to show

        else:

            # Show data from network (activations or diffs)
            if self.state.layers_show_back:
                back_what_to_disp = self.get_back_what_to_disp()
                if back_what_to_disp == 'disabled':
                    layer_dat_3D_normalized = np.tile(self.settings.window_background, layer_dat_3D.shape + (1,))
                elif back_what_to_disp == 'stale':
                    layer_dat_3D_normalized = np.tile(self.settings.stale_background, layer_dat_3D.shape + (1,))
                else:
                    layer_dat_3D_normalized = tile_images_normalize(layer_dat_3D,
                                                                    boost_indiv=self.state.layer_boost_indiv,
                                                                    boost_gamma=self.state.layer_boost_gamma,
                                                                    neg_pos_colors=((1, 0, 0), (0, 1, 0)))
            else:
                layer_dat_3D_normalized = tile_images_normalize(layer_dat_3D,
                                                                boost_indiv=self.state.layer_boost_indiv,
                                                                boost_gamma=self.state.layer_boost_gamma)
            # print ' ===layer_dat_3D_normalized.shape', layer_dat_3D_normalized.shape, 'layer_dat_3D_normalized dtype', layer_dat_3D_normalized.dtype, 'range', layer_dat_3D_normalized.min(), layer_dat_3D_normalized.max()

            if state_layers_pane_filter_mode in (0, 1, 2):
                display_3D = layer_dat_3D_normalized

        # Convert to float if necessary:
        display_3D = ensure_float01(display_3D)

        # Upsample gray -> color if necessary
        #   e.g. (1000,32,32) -> (1000,32,32,3)
        if len(display_3D.shape) == 3:
            display_3D = display_3D[:, :, :, np.newaxis]

        if display_3D.shape[3] == 1:
            display_3D = np.tile(display_3D, (1, 1, 1, 3))
        # Upsample unit length tiles to give a more sane tile / highlight ratio
        #   e.g. (1000,1,1,3) -> (1000,3,3,3)
        if display_3D.shape[1] == 1:
            display_3D = np.tile(display_3D, (1, 3, 3, 1))

        if state_layers_pane_zoom_mode in (0, 2):

            highlights = [None] * n_tiles
            with self.state.lock:
                if self.state.cursor_area == 'bottom':
                    highlights[self.state.selected_unit] = self.settings.kerasvis_layer_clr_cursor  # in [0,1] range
                if self.state.backprop_selection_frozen and self.state.layer == self.state.backprop_layer:
                    highlights[self.state.backprop_unit] = self.settings.kerasvis_layer_clr_back_sel  # in [0,1] range

            if self.state.cursor_area == 'bottom' and state_layers_pane_filter_mode in (3, 4):
                # pane.data[0:display_2D_resize.shape[0], 0:2, :] = to_255(self.settings.window_background)
                # pane.data[0:2, 0:display_2D_resize.shape[1], :] = to_255(self.settings.window_background)
                display_3D[self.state.selected_unit, 0:display_3D.shape[1], 0:2,
                :] = self.settings.kerasvis_layer_clr_cursor
                display_3D[self.state.selected_unit, 0:2, 0:display_3D.shape[2],
                :] = self.settings.kerasvis_layer_clr_cursor

                display_3D[self.state.selected_unit, 0:display_3D.shape[1], -2:,
                :] = self.settings.kerasvis_layer_clr_cursor
                display_3D[self.state.selected_unit, -2:, 0:display_3D.shape[2],
                :] = self.settings.kerasvis_layer_clr_cursor

            _, display_2D = tile_images_make_tiles(display_3D, hw=(tile_rows, tile_cols), padval=padval,
                                                   highlights=highlights)

            # Mode 0: normal display (activations or patterns)
            display_2D_resize = ensure_uint255_and_resize_to_fit(display_2D, pane.data.shape)
            if state_layers_pane_zoom_mode == 2:
                display_2D_resize = display_2D_resize * 0

            if display_3D_highres is None:
                display_3D_highres = display_3D

        elif state_layers_pane_zoom_mode == 1:
            if display_3D_highres is None:
                display_3D_highres = display_3D

            # Mode 1: zoomed selection
            if state_layers_pane_filter_mode in (0, 1, 2):
                unit_data = display_3D_highres[self.state.selected_unit]
            else:
                unit_data = display_3D_highres[0]

            display_2D_resize = ensure_uint255_and_resize_to_fit(unit_data, pane.data.shape)

        pane.data[:] = to_255(self.settings.window_background)
        pane.data[0:display_2D_resize.shape[0], 0:display_2D_resize.shape[1], :] = display_2D_resize

        # # Add background strip around the top and left edges
        # pane.data[0:display_2D_resize.shape[0], 0:2, :] = to_255(self.settings.window_background)
        # pane.data[0:2, 0:display_2D_resize.shape[1], :] = to_255(self.settings.window_background)

        if self.settings.kerasvis_label_layers and \
                self.state.layer in self.settings.kerasvis_label_layers and \
                self.labels and self.state.cursor_area == 'bottom':
            # Display label annotation atop layers pane (e.g. for fc8/prob)
            defaults = {'face': getattr(cv2, self.settings.kerasvis_label_face),
                        'fsize': self.settings.kerasvis_label_fsize,
                        'clr': to_255(self.settings.kerasvis_label_clr),
                        'thick': self.settings.kerasvis_label_thick}
            loc_base = self.settings.kerasvis_label_loc[::-1]  # Reverse to OpenCV c,r order
            lines = [FormattedString(self.labels[self.state.selected_unit], defaults)]
            cv2_typeset_text(pane.data, lines, loc_base)

        return display_3D_highres, selected_unit_highres

    def _draw_selected_pane(self, pane, layer_data_normalized, selected_unit_highres=None):
        pane.data[:] = to_255(self.settings.window_background)

        with self.state.lock:
            mode = 'selected' if self.state.cursor_area == 'bottom' else 'none'

        if mode == 'selected':
            unit_data = None
            if selected_unit_highres is not None:
                unit_data = selected_unit_highres
            else:
                if self.state.selected_unit < len(layer_data_normalized):
                    unit_data = layer_data_normalized[self.state.selected_unit]
                elif len(layer_data_normalized) == 1:
                    unit_data = layer_data_normalized[0]

            if unit_data is not None:
                unit_data_resize = ensure_uint255_and_resize_to_fit(unit_data, pane.data.shape)
                pane.data[0:unit_data_resize.shape[0], 0:unit_data_resize.shape[1], :] = unit_data_resize

    def _draw_aux_pane(self, pane, layer_data_normalized, selected_unit_highres=None):
        pane.data[:] = to_255(self.settings.window_background)

        with self.state.lock:
            if self.state.layers_pane_zoom_mode == 1:
                mode = 'prob_labels'
            elif self.state.cursor_area == 'bottom' and layer_data_normalized is not None:
                mode = 'selected'
            elif self.state.layers_pane_filter_mode in (0, 1, 2, 3):
                mode = 'prob_labels'
            else:
                mode = 'none'

        # if mode == 'selected' and layer_data_normalized is None and selected_unit_highres is None:
        #     mode = 'prob_labels'

        if mode == 'selected':
            if selected_unit_highres is not None:
                unit_data = selected_unit_highres
            else:
                unit_data = layer_data_normalized[self.state.selected_unit]
            unit_data_resize = ensure_uint255_and_resize_to_fit(unit_data, pane.data.shape)
            pane.data[0:unit_data_resize.shape[0], 0:unit_data_resize.shape[1], :] = unit_data_resize
        elif mode == 'prob_labels':
            self._draw_prob_labels_pane(pane)

    def _draw_back_pane(self, pane):
        with self.state.lock:
            back_mode = self.state.back_mode
            back_filt_mode = self.state.back_filt_mode
            back_what_to_disp = self.get_back_what_to_disp()

        if back_what_to_disp == 'disabled':
            pane.data[:] = to_255(self.settings.window_background)

        elif back_what_to_disp == 'stale':
            pane.data[:] = to_255(self.settings.stale_background)

        else:
            # One of the backprop modes is enabled and the back computation (gradient or deconv) is up to date

            grad_blob = self.net.blobs['data'].diff

            # Manually deprocess (skip mean subtraction and rescaling)
            # grad_img = self.net.deprocess('data', diff_blob)
            grad_blob = grad_blob[0]  # bc01 -> c01
            grad_blob = grad_blob.transpose((1, 2, 0))  # c01 -> 01c
            grad_img = grad_blob[:, :, self._net_channel_swap_inv]  # e.g. BGR -> RGB

            # Mode-specific processing
            assert back_mode in ('grad', 'deconv')
            assert back_filt_mode in ('raw', 'gray', 'norm', 'normblur')
            if back_filt_mode == 'raw':
                grad_img = norm01c(grad_img, 0)
            elif back_filt_mode == 'gray':
                grad_img = grad_img.mean(axis=2)
                grad_img = norm01c(grad_img, 0)
            elif back_filt_mode == 'norm':
                grad_img = np.linalg.norm(grad_img, axis=2)
                grad_img = norm01(grad_img)
            else:
                grad_img = np.linalg.norm(grad_img, axis=2)
                cv2.GaussianBlur(grad_img, (0, 0), self.settings.kerasvis_grad_norm_blur_radius, grad_img)
                grad_img = norm01(grad_img)

            # If necessary, re-promote from grayscale to color
            if len(grad_img.shape) == 2:
                grad_img = np.tile(grad_img[:, :, np.newaxis], 3)

            grad_img_resize = ensure_uint255_and_resize_to_fit(grad_img, pane.data.shape)

            pane.data[0:grad_img_resize.shape[0], 0:grad_img_resize.shape[1], :] = grad_img_resize

    def _draw_jpgvis_pane(self, pane):
        pane.data[:] = to_255(self.settings.window_background)

        with self.state.lock:
            state_layer, state_selected_unit, cursor_area, show_unit_jpgs = self.state.layer, self.state.selected_unit, self.state.cursor_area, self.state.show_unit_jpgs

        try:
            # Some may be missing this setting
            self.settings.kerasvis_jpgvis_layers
        except:
            print '\n\nNOTE: you need to upgrade your settings.py and settings_local.py files. See README.md.\n\n'
            raise

        if self.settings.kerasvis_jpgvis_remap and state_layer in self.settings.kerasvis_jpgvis_remap:
            img_key_layer = self.settings.kerasvis_jpgvis_remap[state_layer]
        else:
            img_key_layer = state_layer

        if self.settings.kerasvis_jpgvis_layers and img_key_layer in self.settings.kerasvis_jpgvis_layers and cursor_area == 'bottom' and show_unit_jpgs:
            img_key = (img_key_layer, state_selected_unit, pane.data.shape)
            img_resize = self.img_cache.get(img_key, None)
            if img_resize is None:
                # If img_resize is None, loading has not yet been attempted, so show stale image and request load by JPGVisLoadingThread
                with self.state.lock:
                    self.state.jpgvis_to_load_key = img_key
                pane.data[:] = to_255(self.settings.stale_background)
            elif img_resize.nbytes == 0:
                # This is the sentinal value when the image is not
                # found, i.e. loading was already attempted but no jpg
                # assets were found. Just display disabled.
                pane.data[:] = to_255(self.settings.window_background)
            else:
                # Show image
                pane.data[:img_resize.shape[0], :img_resize.shape[1], :] = img_resize
        else:
            # Will never be available
            pane.data[:] = to_255(self.settings.window_background)

    def handle_key(self, key, panes):
        return self.state.handle_key(key)

    def get_back_what_to_disp(self):
        '''Whether to show back diff information or stale or disabled indicator'''
        if (
                self.state.cursor_area == 'top' and not self.state.backprop_selection_frozen) or not self.state.back_enabled:
            return 'disabled'
        elif self.state.back_stale:
            return 'stale'
        else:
            return 'normal'

    def set_debug(self, level):
        self.debug_level = level
        self.proc_thread.debug_level = level
        self.jpgvis_thread.debug_level = level

    def draw_help(self, help_pane, locy):
        defaults = {'face': getattr(cv2, self.settings.help_face),
                    'fsize': self.settings.help_fsize,
                    'clr': to_255(self.settings.help_clr),
                    'thick': self.settings.help_thick}
        loc_base = self.settings.help_loc[::-1]  # Reverse to OpenCV c,r order
        locx = loc_base[0]

        lines = []
        lines.append([FormattedString('', defaults)])
        lines.append([FormattedString('Kerasvis keys', defaults)])

        kl, _ = self.bindings.get_key_help('sel_left')
        kr, _ = self.bindings.get_key_help('sel_right')
        ku, _ = self.bindings.get_key_help('sel_up')
        kd, _ = self.bindings.get_key_help('sel_down')
        klf, _ = self.bindings.get_key_help('sel_left_fast')
        krf, _ = self.bindings.get_key_help('sel_right_fast')
        kuf, _ = self.bindings.get_key_help('sel_up_fast')
        kdf, _ = self.bindings.get_key_help('sel_down_fast')

        keys_nav_0 = ','.join([kk[0] for kk in (kl, kr, ku, kd)])
        keys_nav_1 = ''
        if len(kl) > 1 and len(kr) > 1 and len(ku) > 1 and len(kd) > 1:
            keys_nav_1 += ' or '
            keys_nav_1 += ','.join([kk[1] for kk in (kl, kr, ku, kd)])
        keys_nav_f = ','.join([kk[0] for kk in (klf, krf, kuf, kdf)])
        nav_string = 'Navigate with %s%s. Use %s to move faster.' % (keys_nav_0, keys_nav_1, keys_nav_f)
        lines.append([FormattedString('', defaults, width=120, align='right'),
                      FormattedString(nav_string, defaults)])

        for tag in ('sel_layer_left', 'sel_layer_right', 'log_scale', 'zoom_mode',
                    'filter_mode', 'pattern_mode', 'ez_back_mode_loop', 'freeze_back_unit',
                    'show_back', 'back_mode', 'back_filt_mode', 'boost_gamma', 'boost_individual',
                    'reset_state'):
            key_strings, help_string = self.bindings.get_key_help(tag)
            label = '%10s:' % (','.join(key_strings))
            lines.append([FormattedString(label, defaults, width=120, align='right'),
                          FormattedString(help_string, defaults)])

        locy = cv2_typeset_text(help_pane.data, lines, (locx, locy),
                                line_spacing=self.settings.help_line_spacing)

        return locy

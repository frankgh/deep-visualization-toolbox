import os
import re
import time
import timeit
from threading import RLock

import numpy as np

from codependent_thread import CodependentThread
from image_misc import plt_plot_signal, crop_to_square


class InputSignalFetcher(CodependentThread):
    '''Fetches signals from a directory.'''

    def __init__(self, settings):
        CodependentThread.__init__(self, settings.input_updater_heartbeat_required)
        self.daemon = True
        self.lock = RLock()
        self.quit = False
        self.debug_level = 0
        self.latest_frame_idx = -1
        self.latest_signal_idx = -1
        self.latest_frame_data = None
        self.latest_signal = None

        self.static_file_mode = True
        self.settings = settings
        self.static_file_stretch_mode = self.settings.static_file_stretch_mode
        self.sleep_after_read_frame = settings.input_updater_sleep_after_read_frame

        self.signal_apply_filter = False
        self.signal_zoom_level = 1.0
        self.signal_offset = 0

        # Static file input
        self.latest_static_filename = None
        self.latest_static_file_data = None
        self.latest_static_frame = None
        self.latest_label = None
        self.static_file_idx = None
        self.static_file_idx_increment = 0
        self.signal_idx = None
        self.last_signal_idx = None
        self.signal_idx_increment = 0
        self.signal_labels = settings.signal_labels if hasattr(settings, 'signal_labels') else None

    def set_debug(self, debug_level):
        with self.lock:
            self.debug_level = debug_level

    def set_mode_static(self):
        with self.lock:
            self.static_file_mode = True

    def set_mode_stretch_on(self):
        with self.lock:
            if not self.static_file_stretch_mode:
                self.static_file_stretch_mode = True
                self.latest_static_frame = None  # Force reload

    def set_mode_stretch_off(self):
        with self.lock:
            if self.static_file_stretch_mode:
                self.static_file_stretch_mode = False
                self.latest_static_frame = None  # Force reload

    def toggle_stretch_mode(self):
        with self.lock:
            if self.static_file_stretch_mode:
                self.set_mode_stretch_off()
            else:
                self.set_mode_stretch_on()

    def run(self):
        while not self.quit and not self.is_timed_out():
            # start_time = time.time()
            if self.static_file_mode:
                self.check_increment_and_load_image()
                pass
            else:
                print 'Only static_file_mode is supported'

            time.sleep(self.sleep_after_read_frame)
            # print 'Reading one frame took', time.time() - start_time

        print 'InputSignalFetcher: exiting run method'
        # print 'InputSignalFetcher: read', self.read_frames, 'frames'

    def get_frame(self):
        '''Fetch the latest frame_idx and frame. The idx increments
        any time the frame data changes. If the idx is < 0, the frame
        is not valid.
        '''
        with self.lock:
            return (self.latest_frame_idx, self.latest_frame_data, self.latest_signal_idx, self.latest_signal,
                    self.latest_label)

    def increment_static_file_idx(self, amount=1):
        with self.lock:
            self.static_file_idx_increment += amount

    def toggle_filter(self):
        with self.lock:
            self.signal_apply_filter = not self.signal_apply_filter
            print ('toggle_filter', self.signal_apply_filter )
            sig = self._plot()
            self._increment_and_set_frame(self.latest_static_frame, sig)

    def increment_zoom_level(self, amount=0.05):
        with self.lock:
            curr = self.signal_zoom_level
            self.signal_zoom_level = min(1.0, max(0.01, self.signal_zoom_level + amount))

            if curr != self.signal_zoom_level:
                self._plot()
                self._increment_and_set_frame(self.latest_static_frame, None)

    def move_signal(self, amount=100):
        with self.lock:
            new_value = max(0, self.signal_offset + amount)

            if self._plot():
                self.signal_offset = new_value
                self._plot()
                self._increment_and_set_frame(self.latest_static_frame, None)

    def increment_signal_idx(self, amount=1):
        with self.lock:
            self.signal_idx_increment += amount

    def _increment_and_set_frame(self, frame, signal):
        with self.lock:
            if frame is not None:
                self.latest_frame_idx += 1
                self.latest_frame_data = frame
            if signal is not None:
                self.latest_signal_idx += 1
                self.latest_signal = signal

    def check_increment_and_load_image(self):
        with self.lock:

            if (self.static_file_idx_increment == 0
                    and self.static_file_idx is not None
                    and self.signal_idx_increment == 0
                    and self.signal_idx is not None
                    and self.latest_static_frame is not None):
                return  # Skip if a static frame is already loaded and there is no increment

            match_flags = re.IGNORECASE if self.settings.static_files_ignore_case else 0
            available_files = [filename for filename in os.listdir(self.settings.static_files_dir) if
                               re.match(self.settings.static_files_regexp, filename, match_flags)]

            assert len(
                available_files) != 0, 'Error: No files found in {} matching {} (current working directory is {})'.format(
                self.settings.static_files_dir, self.settings.static_files_regexp, os.getcwd())

            if self.debug_level == 3:
                print 'Found files:'
                for filename in available_files:
                    print '   {}'.format(filename)

            if self.static_file_idx is None:
                self.static_file_idx = 0

            self.static_file_idx = (self.static_file_idx + self.static_file_idx_increment) % len(available_files)
            self.static_file_idx_increment = 0
            new_data_file_loaded = False

            assert hasattr(self.settings,
                           'static_files_data_fn'), 'Error: Your settings_local.py does not define a static_files_data_fn function'
            assert hasattr(self.settings,
                           'static_files_labels_fn'), 'Error: Your settings_local.py does not define a static_files_labels_fn function'

            if self.latest_static_file_data is None or self.latest_static_filename != available_files[
                self.static_file_idx]:
                self.latest_static_filename = available_files[self.static_file_idx]
                self.latest_static_file = np.load(
                    os.path.join(self.settings.static_files_dir, self.latest_static_filename))
                self.latest_static_file_labels = self.settings.static_files_labels_fn(self.latest_static_file)
                self.latest_static_file_data = self.settings.static_files_data_fn(self.latest_static_file)

                self.signal_idx = 0
                new_data_file_loaded = True

            if self.signal_idx is None:
                self.signal_idx = 0

            available_signal_count = self.latest_static_file_labels.shape[0]

            self.signal_idx = (self.signal_idx + self.signal_idx_increment) % available_signal_count
            self.signal_idx_increment = 0

            if new_data_file_loaded or self.latest_static_frame is None or self.last_signal_idx is None or self.last_signal_idx != self.signal_idx:
                self.last_signal_idx = self.signal_idx
                self._plot()

            self._increment_and_set_frame(self.latest_static_frame,
                                          self.latest_static_file_data[self.signal_idx:self.signal_idx + 1])

    def _plot(self):
        markers = None
        start_time = timeit.default_timer()
        sig = self.latest_static_file_data[self.signal_idx]
        if self.signal_apply_filter and hasattr(self.settings, 'signal_filter_fn'):
            sig, markers = self.settings.signal_filter_fn(sig)
        im = plt_plot_signal(sig, self.signal_labels, zoom_level=self.signal_zoom_level, markers=markers)
        elapsed = timeit.default_timer() - start_time
        print('plt_plot_signal function ran for', elapsed)

        if not self.static_file_stretch_mode:
            im = crop_to_square(im)
        self.latest_static_frame = im
        self.latest_label = self.latest_static_file_labels[self.signal_idx]
        return sig

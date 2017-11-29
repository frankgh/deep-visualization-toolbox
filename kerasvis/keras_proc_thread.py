import time
import timeit

from keras import backend as K

from codependent_thread import CodependentThread
from misc import WithTimer


class KerasProcThread(CodependentThread):
    '''Runs Keras in separate thread.'''

    def __init__(self, net, graph, state, loop_sleep, pause_after_keys, heartbeat_required):
        CodependentThread.__init__(self, heartbeat_required)
        self.daemon = True
        self.net = net
        self.graph = graph
        self.state = state
        self.last_process_finished_at = None
        self.last_process_elapsed = None
        self.frames_processed_fwd = 0
        self.frames_processed_back = 0
        self.loop_sleep = loop_sleep
        self.pause_after_keys = pause_after_keys
        self.debug_level = 0
        self.predictions = None

    def get_predictions(self):
        return self.predictions

    def run(self):
        print 'KerasProcThread.run called'

        while not self.is_timed_out():
            with self.state.lock:
                if self.state.quit:

                    if self.debug_level == 3:
                        print 'KerasProcThread.run: quit is: {}'.format(self.state.quit)

                    break

                if self.debug_level == 3:
                    print 'KerasProcThread.run: keras_net_state is: {}'.format(self.state.keras_net_state)

                    print 'KerasProcThread.run loop: next_frame: {}, keras_net_state: {}, back_enabled: {}'.format(
                        'None' if self.state.next_frame is None else 'Avail',
                        self.state.keras_net_state,
                        self.state.back_enabled)

                frame = None
                run_fwd = False
                run_back = False
                if self.state.keras_net_state == 'free' and time.time() - self.state.last_key_at > self.pause_after_keys:
                    frame = self.state.next_frame
                    self.state.next_frame = None
                    back_enabled = self.state.back_enabled
                    back_mode = self.state.back_mode
                    back_stale = self.state.back_stale
                    # state_layer = self.state.layer
                    # selected_unit = self.state.selected_unit
                    backprop_layer = self.state.backprop_layer
                    backprop_unit = self.state.backprop_unit

                    # Forward should be run for every new frame
                    run_fwd = (frame is not None)
                    # Backward should be run if back_enabled and (there was a new frame OR back is stale (new backprop layer/unit selected))
                    run_back = (back_enabled and (run_fwd or back_stale))
                    self.state.keras_net_state = 'proc' if (run_fwd or run_back) else 'free'

            if self.debug_level == 3:
                print 'run_fwd = {}, run_back = {}'.format(run_fwd, run_back)

            if run_fwd:
                if self.debug_level == 3:
                    print 'TIMING:, processing frame'
                self.frames_processed_fwd += 1

                with WithTimer('KerasProcThread:forward', quiet=self.debug_level < 1):
                    with self.graph.as_default():
                        start_time = timeit.default_timer()
                        # intermediate_layer_model = Model(inputs=self.net.input,
                        #                                  outputs=self.net.layers[self.state.layer_idx].output)
                        # final_layer_model = Model(inputs=self.net.layers[self.state.layer_idx+1].input,
                        #                           outputs=self.net.layers[-1].output)
                        # intermediate_output = intermediate_layer_model.predict(frame, verbose=self.debug_level)
                        # final_output = final_layer_model.predict(intermediate_output, verbose=self.debug_level)

                        inp = self.net.input
                        outputs = [self.net.layers[self.state.layer_idx].output, self.net.layers[-1].output]
                        functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
                        layer_outs = functor([frame, 0.])

                        self.intermediate_predictions = layer_outs[0][0]
                        self.predictions = layer_outs[1][0]
                        # self.predictions = self.net.predict(frame, verbose=self.debug_level)[0]
                        elapsed = timeit.default_timer() - start_time
                        print('self.net.predict function ran for', elapsed)

                    if self.debug_level == 3:
                        print ('KerasProcThread:forward self.net.predict:', self.predictions)

            if run_back:
                diffs = self.net.blobs[backprop_layer].diff * 0
                diffs[0][backprop_unit] = self.net.blobs[backprop_layer].data[0, backprop_unit]

                assert back_mode in ('grad', 'deconv')
                if back_mode == 'grad':
                    with WithTimer('KerasProcThread:backward', quiet=self.debug_level < 1):
                        if self.debug_level == 3:
                            print '**** Doing backprop with {} diffs in [{},{}]'.format(backprop_layer, diffs.min(),
                                                                                        diffs.max())
                        try:
                            self.net.backward_from_layer(backprop_layer, diffs, zero_higher=True)
                        except AttributeError:
                            print 'ERROR: required bindings (backward_from_layer) not found! Try using the deconv-deep-vis-toolbox branch as described here: https://github.com/yosinski/deep-visualization-toolbox'
                            raise
                else:
                    with WithTimer('KerasProcThread:deconv', quiet=self.debug_level < 1):
                        if self.debug_level == 3:
                            print '**** Doing deconv with {} diffs in [{},{}]'.format(backprop_layer, diffs.min(),
                                                                                      diffs.max())
                        try:
                            self.net.deconv_from_layer(backprop_layer, diffs, zero_higher=True)
                        except AttributeError:
                            print 'ERROR: required bindings (deconv_from_layer) not found! Try using the deconv-deep-vis-toolbox branch as described here: https://github.com/yosinski/deep-visualization-toolbox'
                            raise

                with self.state.lock:
                    self.state.back_stale = False

            if run_fwd or run_back:
                with self.state.lock:
                    self.state.keras_net_state = 'free'
                    self.state.drawing_stale = True
                now = time.time()
                if self.last_process_finished_at:
                    self.last_process_elapsed = now - self.last_process_finished_at
                self.last_process_finished_at = now
            else:
                time.sleep(self.loop_sleep)

        print 'KerasProcThread.run: finished'
        print 'KerasProcThread.run: processed {} frames fwd, {} frames back'.format(self.frames_processed_fwd,
                                                                                    self.frames_processed_back)

    def approx_fps(self):
        '''Get the approximate frames per second processed by this
        thread, considering only the last signal processed. If more
        than two seconds ago, assume pipeline has stalled elsewhere
        (perhaps using static signals that are only processed once).
        '''
        if self.last_process_elapsed and (time.time() - self.last_process_finished_at) < 2.0:
            return 1.0 / (self.last_process_elapsed + 1e-6)
        else:
            return 0.0

#! /usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from bindings import bindings
from signal_vis import SignalVis

try:
    import settings
except:
    print '\nError importing settings.py. Check the error message below for more information.'
    print 'If you haven\'t already, you\'ll want to copy the settings_local.template-*.py files'
    print 'to settings_local.py.'
    print
    print ' $ cp models/kerasvis-afg/settings_local.template-kerasvis-afg.py settings_local.py'
    raise


def main():
    sv = SignalVis(settings)

    help_keys, _ = bindings.get_key_help('help_mode')
    quit_keys, _ = bindings.get_key_help('quit')
    print '\n\nRunning toolbox. Push {} for help or {} to quit'.format(help_keys[0], quit_keys[0])
    sv.run_loop()


if __name__ == '__main__':
    main()

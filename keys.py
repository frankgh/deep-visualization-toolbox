# Define keys

# class KeyPatten(object):
#    '''Define a pattern that will be matched against a keycode.
#
#    A KeyPattern is used to determine which key was pressed in
#    OpenCV. This process is complicated by the fact that different
#    platforms define different key codes for each key. Further, on
#    some platforms the value returned by OpenCV is different than that
#    returned by Python ord(). See the following link for more
#    information:
#    https://stackoverflow.com/questions/14494101/using-other-keys-for-the-waitkey-function-of-opencv/20577067#20577067
#    '''
#    def __init__(self, code, mask = None):
#        self.code = code
#        self.mask = mask
#        #self.mask = 0xffffffff    # 64 bits. All codes observed so far are < 2**64



# Larger masks (requiring a more specific pattern) are matched first
key_data = []
for letter in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
    # for letter in 'abefghijklmnopqrstuvwxyzABEFGHIJKLMNOPQRSTUVWXYZ01456789':
    key_data.append((letter, ord(letter), 0xff))  # Match only lowest byte

key_data.extend([
    # Mac (note diff order vs Linux)
    ('up', 0xf700, 0xffff),
    ('down', 0xf701, 0xffff),
    ('left', 0xf702, 0xffff),
    ('right', 0xf703, 0xffff),
    ('pgup', 0xf72c, 0xffff),
    ('pgdn', 0xf72d, 0xffff),

    # Ubuntu US/UK (note diff order vs Mac)
    ('left', 0xff51, 0xffff),
    ('up', 0xff52, 0xffff),
    ('right', 0xff53, 0xffff),
    ('down', 0xff54, 0xffff),

    # Ubuntu only; modified keys to not produce separate events on
    # Mac. These are included only so they be ignored without
    # producing error messages.
    ('leftshift', 0xffe1, 0xffff),
    ('rightshift', 0xffe2, 0xffff),
    ('leftctrl', 0xffe3, 0xffff),
    ('rightctrl', 0xffe4, 0xffff),
    ('esc', 27, 0xff),  # Mac
    ('enter', 13, 0xff),  # Mac
    ('enter', 10, 0xff),  # Ubuntu with UK keyboard
])

key_patterns = dict()
# Store key_patterns by mask in a dict of dicts
# Eventually, e.g.:
#   key_patterns[0xff][97] = 'a'
for key_datum in key_data:
    # print key_datum
    assert len(key_datum) in (2, 3), 'Key information should be tuple of length 2 or 3 but it is %s' % repr(key_datum)
    if len(key_datum) == 3:
        label, key_code, mask = key_datum
    else:
        label, key_code = key_datum
        mask = 0xffffffff  # 64 bits. All codes observed so far are < 2**64
    if not mask in key_patterns:
        key_patterns[mask] = dict()
    if key_code in key_patterns[mask]:
        old_label = key_patterns[mask][key_code]
        if old_label != label:
            print 'Warning: key_patterns[%s][%s] old value %s being overwritten with %s' % (
                mask, key_code, old_label, label)
    if key_code != (key_code & mask):
        print 'Warning: key_code %s for key label %s will never trigger using mask %s' % (key_code, label, mask)
    key_patterns[mask][key_code] = label

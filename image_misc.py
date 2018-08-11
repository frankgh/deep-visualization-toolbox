#! /usr/bin/env python

import cv2
import matplotlib.pyplot as plt
import skimage
import skimage.io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import arange, array, newaxis, tile, linspace, pad, expand_dims, \
    fromstring, ceil, dtype, float32, sqrt, dot, zeros

from misc import WithTimer


def norm01(arr):
    arr = arr.copy()
    arr -= arr.min()
    arr /= arr.max() + 1e-10
    return arr


def norm01c(arr, center):
    '''Maps the input range to [0,1] such that the center value maps to .5'''
    arr = arr.copy()
    arr -= center
    arr /= max(2 * arr.max(), -2 * arr.min()) + 1e-10
    arr += .5
    assert arr.min() >= 0
    assert arr.max() <= 1
    return arr


def norm0255(arr):
    '''Maps the input range to [0,255] as dtype uint8'''
    arr = arr.copy()
    arr -= arr.min()
    arr *= 255.0 / (arr.max() + 1e-10)
    arr = array(arr, 'uint8')
    return arr


def cv2_read_cap_rgb(cap, saveto=None):
    rval, frame = cap.read()
    if saveto:
        cv2.imwrite(saveto, frame)
    if len(frame.shape) == 2:
        # Upconvert single channel grayscale to color
        frame = frame[:, :, newaxis]
    if frame.shape[2] == 1:
        frame = tile(frame, (1, 1, 3))
    if frame.shape[2] > 3:
        # Chop off transparency
        frame = frame[:, :, :3]
    frame = frame[:, :, ::-1]  # Convert native OpenCV BGR -> RGB
    return frame


def plt_plot_signal(data, labels, zoom_level=-1, offset=0, markers=None, title=None):
    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = None

    if len(data.shape) == 1:
        data = expand_dims(data, axis=1)

    if zoom_level == -1:
        zoom_level = data.shape[0]

    color = iter(cm.rainbow(linspace(0, 1, data.shape[1])))

    s = offset
    e = s + zoom_level
    x = arange(s, e)

    for i in range(data.shape[1]):
        c = next(color)
        label = labels[i] if labels is not None else 'Signal {}'.format(i + 1)
        ax = fig.add_subplot(data.shape[1], 1, (i + 1), sharex=ax)
        ax.plot(x, data[s:e, i], lw=1, label=label, c=c)
        # # ax.set_adjustable('box-forced')
        # ax.set_xlim(left=0, right=zoom_level)
        # ax.get_xaxis().set_visible(i == data.shape[1] - 1)

        # ax.xaxis.set_ticks(arange(s, e + 1, (e - s) / 10.0))
        # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        ax.legend(loc='lower right')

        if markers is not None and i in markers:
            for val in markers[i]:
                if val >= s and val < e:
                    ax.axvline(x=val)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    canvas.draw()  # draw the canvas, cache the renderer

    l, b, w, h = fig.bbox.bounds
    w, h = int(w), int(h)

    im = fromstring(canvas.tostring_rgb(), dtype='uint8')
    im.shape = h, w, 3
    return im


def plt_plot_heatmap(data,
                     shape,
                     rows,
                     cols,
                     title=None,
                     x_axis_label=None,
                     y_axis_label=None,
                     x_axis_values=None,
                     y_axis_values=None,
                     hide_axis=True,
                     vmin=None,
                     vmax=None):
    """
    Most ideas were taken
    from https://stackoverflow.com/questions/45697522/seaborn-heatmap-plotting-execution-time-optimization
    """
    res = []
    shape = (max(2, ceil(shape[1] / 80 / cols)), max(2, ceil(shape[0] / 80 / rows)))
    fig, ax = plt.subplots(1, 1, figsize=shape)
    canvas = FigureCanvas(fig)

    # for i in xrange(y.shape[0]):
    #     sns.heatmap(y[i], ax=ax, vmin=minn, vmax=maxx)
    #     canvas.draw()  # draw the canvas, cache the renderer
    #
    #     l, b, w, h = fig.bbox.bounds
    #     w, h = int(w), int(h)
    #     im = fromstring(canvas.tostring_rgb(), dtype='uint8')
    #     im.shape = h, w, 3
    #     res.append(im)

    img = ax.imshow(
        zeros((data.shape[1], data.shape[2])),
        cmap='viridis',
        vmin=vmin if vmin is not None else data.min(),
        vmax=vmax if vmax is not None else data.max(),
        interpolation='none',
        aspect='auto'
    )

    # get rid of spines and fix range of axes, rotate x-axis labels
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if hide_axis:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0, wspace=0)
    else:

        if title is not None:
            plt.title(title)

        if x_axis_label is not None:
            ax.set_xlabel(x_axis_label)

        if y_axis_label is not None:
            ax.set_ylabel(y_axis_label)

        if x_axis_values is not None:
            a = arange(0, x_axis_values.shape[0], 3) + 0.5
            b = arange(x_axis_values.min(), x_axis_values.max() + 1.5, 1.5)
            ax.set_xticks(a)
            ax.set_xticklabels(b, rotation=90)

        if y_axis_values is not None:
            a = arange(0, y_axis_values.shape[0], 3) + 0.5
            # c = roundup((y_axis_values.max() - y_axis_values.min()) / 11)
            # b = arange(y_axis_values.min(), y_axis_values.max(), c)
            b = linspace(y_axis_values.min(), y_axis_values.max(), num=10, dtype=int)
            ax.set_yticks(a)
            ax.set_yticklabels(b)

    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(90)

    if not hide_axis:
        divider = make_axes_locatable(ax)
        # colorbar on the right of ax. Colorbar width in % of ax and space between them is defined by pad in inches
        cax = divider.append_axes('right', size='5%', pad=0.07)
        cb = fig.colorbar(img, cax=cax)
        # remove colorbar frame/spines
        cb.outline.set_visible(False)

    # don't stop after each subfigure change
    plt.show(block=False)

    if not hide_axis:
        fig.tight_layout()
    canvas.draw()  # draw the canvas, cache the renderer

    # keep bg in memory
    background = fig.canvas.copy_from_bbox(ax.bbox)
    # start = time.time()
    for i in xrange(data.shape[0]):
        img.set_array(data[i])

        # restore background
        fig.canvas.restore_region(background)

        ax.draw_artist(img)

        # fill in the axes rectangle
        fig.canvas.blit(ax.bbox)

        # loop through array

        # for i in xrange(data.shape[0]):
        #     time.sleep(0.005)
        #     img.set_array(data[i])
        #     canvas.draw()

        l, b, w, h = fig.bbox.bounds
        w, h = int(w), int(h)
        im = fromstring(canvas.tostring_rgb(), dtype='uint8')
        im.shape = h, w, 3
        res.append(im)

    fig.clf()
    plt.clf()
    plt.close()
    return array(res)


def plt_plot_filter(x, y, title, x_axis_label, y_axis_label, log_scale):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    canvas = FigureCanvas(fig)
    x = arange(0, y.shape[0]) if x is None else x
    if log_scale == 1:
        ax.semilogy(x, y, lw=2)
    else:
        ax.plot(x, y, lw=2)
    ax.set(xlabel=x_axis_label, ylabel=y_axis_label, title=title)

    fig.tight_layout()
    canvas.draw()  # draw the canvas, cache the renderer

    l, b, w, h = fig.bbox.bounds
    w, h = int(w), int(h)
    im = fromstring(canvas.tostring_rgb(), dtype='uint8')
    im.shape = h, w, 3
    fig.clf()
    plt.clf()
    plt.close()
    return im


def plt_plot_filters_blit(y, x, shape, rows, cols,
                          title=None,
                          x_axis_label=None,
                          y_axis_label=None,
                          log_scale=0,
                          hide_axis=False):
    res = []
    x = arange(0, y.shape[1]) if x is None else x

    # if log_scale == 1:
    #     y = log(y)
    # elif log_scale == 2:
    #     x = log(x)
    # elif log_scale == 3:
    #     x = log(x)
    #     y = log(y)
    shape = (max(2, ceil(shape[1] / 80 / cols)), max(2, ceil(shape[0] / 80 / rows)))
    fig, ax = plt.subplots(1, 1, figsize=shape)
    canvas = FigureCanvas(fig)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(y.min(), y.max())

    if hide_axis:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0, wspace=0)
    else:
        if x_axis_label is not None:
            ax.set_xlabel(x_axis_label)

        if y_axis_label is not None:
            ax.set_ylabel(y_axis_label)

    if title is not None:
        plt.title(title)

    line, = ax.plot([], [], lw=2)

    if not hide_axis:
        fig.tight_layout()
    canvas.draw()  # draw the canvas, cache the renderer

    # keep bg in memory
    background = fig.canvas.copy_from_bbox(ax.bbox)

    for i in xrange(y.shape[0]):
        line.set_data(x, y[i])
        # line.set_color()

        # restore background
        fig.canvas.restore_region(background)

        # redraw just the points
        ax.draw_artist(line)

        # fill in the axes rectangle
        fig.canvas.blit(ax.bbox)

        l, b, w, h = fig.bbox.bounds
        w, h = int(w), int(h)
        im = fromstring(canvas.tostring_rgb(), dtype='uint8')
        im.shape = h, w, 3
        res.append(im)

    fig.clf()
    plt.clf()
    plt.close()
    return array(res)


def plt_plot_filters_fast(y, x, shape, rows, cols,
                          title=None,
                          x_axis_label=None,
                          y_axis_label=None,
                          share_axes=True,
                          log_scale=0):
    res = []
    shape = (ceil(shape[1] / 80 / cols), ceil(shape[0] / 80 / rows))
    fig, ax = plt.subplots(1, 1, figsize=shape)
    canvas = FigureCanvas(fig)
    # ax.set_aspect('equal')

    if share_axes:
        if x is not None:
            min_x, max_x = min(x), max(x)
        else:
            min_x, max_x = 0, y.shape[1]
        min_y, max_y = y.min(), y.max()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    # ax.hold(True)
    plt.subplots_adjust(left=0.185, bottom=0.125, right=0.98, top=0.98)
    # plt.show(False)
    # plt.draw()

    # background = fig.canvas.copy_from_bbox(ax.bbox)
    # points = ax.plot(x[0], linewidth=1)[0]

    for i in xrange(y.shape[0]):
        if x is not None:
            if log_scale == 1:
                ax.semilogy(x, y[i], linewidth=1)
            else:
                ax.plot(x, y[i], linewidth=1)
        else:
            if log_scale == 1:
                ax.semilogy(y[i], linewidth=1)
            else:
                ax.plot(y[i], linewidth=1)

        if x_axis_label is not None:
            ax.set_xlabel(x_axis_label)

        if y_axis_label is not None:
            ax.set_ylabel(y_axis_label)

        if title is not None:
            plt.title(title)

        # plt.autoscale(enable=True, axis='y', tight=True)
        # plt.tight_layout()

        # Turn off axes and set axes limits
        # ax.axis('off')

        canvas.draw()  # draw the canvas, cache the renderer

        l, b, w, h = fig.bbox.bounds
        w, h = int(w), int(h)
        im = fromstring(canvas.tostring_rgb(), dtype='uint8')
        im.shape = h, w, 3
        res.append(im)
        # ax.cla()

    fig.clf()
    return array(res)


def plt_plot_filters(x, y, shape, rows, cols,
                     selected_unit=None,
                     selected_unit_color=None,
                     title=None,
                     x_axis_label=None,
                     y_axis_label=None,
                     share_axes=True,
                     log_scale=0):
    shape = (ceil(shape[1] / 80), ceil(shape[0] / 80))
    fig = Figure(figsize=shape)
    canvas = FigureCanvas(fig)
    ax, highlighted_ax, right_ax, bottom_ax, curr, right, bottom = None, None, None, None, None, None, None

    if selected_unit is not None:
        row = selected_unit / cols
        col = selected_unit % cols
        curr = selected_unit
        bottom = (selected_unit + cols) if row < rows - 1 else None
        right = (selected_unit + 1) if col < cols - 1 else None

    for i in xrange(x.shape[0]):
        if share_axes:
            ax = fig.add_subplot(rows, cols, (i + 1), axisbelow=False, sharex=ax, sharey=ax)
        else:
            ax = fig.add_subplot(rows, cols, (i + 1), axisbelow=False)

        if y is not None:
            if log_scale == 1:
                ax.semilogy(y, x[i], linewidth=1)
            else:
                ax.plot(y, x[i], linewidth=1)
        else:
            if log_scale == 1:
                ax.semilogy(x[i], linewidth=1)
            else:
                ax.plot(x[i], linewidth=1)
            ax.set_xlim(left=0, right=x.shape[1] - 1)

        ax.get_xaxis().set_visible(i >= ((rows - 1) * cols))
        ax.get_yaxis().set_visible(i % cols == 0)
        if i == curr:
            highlighted_ax = ax
        if i == bottom:
            bottom_ax = ax
        if i == right:
            right_ax = ax
        if x_axis_label is not None:
            ax.set_xlabel(x_axis_label)
        if y_axis_label is not None:
            ax.set_ylabel(y_axis_label)

    if highlighted_ax is not None:
        for axis in ['top', 'bottom', 'left', 'right']:
            highlighted_ax.spines[axis].set_linewidth(2.5)
            highlighted_ax.spines[axis].set_color(selected_unit_color)

        if bottom_ax is not None:
            bottom_ax.spines['top'].set_linewidth(2)
            bottom_ax.spines['top'].set_color(selected_unit_color)

        if right_ax is not None:
            right_ax.spines['left'].set_linewidth(2)
            right_ax.spines['left'].set_color(selected_unit_color)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    canvas.draw()  # draw the canvas, cache the renderer

    l, b, w, h = fig.bbox.bounds
    w, h = int(w), int(h)
    im = fromstring(canvas.tostring_rgb(), dtype='uint8')
    im.shape = h, w, 3
    return im


def cv2_read_file_rgb(filename):
    '''Reads an image from file. Always returns (x,y,3)'''
    im = cv2.imread(filename)
    if len(im.shape) == 2:
        # Upconvert single channel grayscale to color
        im = im[:, :, newaxis]
    if im.shape[2] == 1:
        im = tile(im, (1, 1, 3))
    if im.shape[2] > 3:
        # Chop off transparency
        im = im[:, :, :3]
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert native OpenCV BGR -> RGB


def crop_to_square(frame):
    i_size, j_size = frame.shape[0], frame.shape[1]
    if j_size > i_size:
        # landscape
        offset = (j_size - i_size) / 2
        return frame[:, offset:offset + i_size, :]
    else:
        # portrait
        offset = (i_size - j_size) / 2
        return frame[offset:offset + j_size, :, :]


def cv2_imshow_rgb(window_name, img):
    # Convert native OpenCV BGR -> RGB before displaying
    cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def caffe_load_image(filename, color=True, as_uint=False):
    '''
    Copied from Caffe to simplify potential import problems.

    Load an image converting from grayscale or alpha as needed.

    Take
    filename: string
    color: flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Give
    image: an image with type float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    '''
    with WithTimer('imread', quiet=True):
        if as_uint:
            img = skimage.io.imread(filename)
        else:
            img = skimage.img_as_float(skimage.io.imread(filename)).astype(float32)
    if img.ndim == 2:
        img = img[:, :, newaxis]
        if color:
            img = tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def get_tiles_height_width(n_tiles, desired_width=None):
    '''Get a height x width size that will fit n_tiles tiles.'''
    if desired_width == None:
        # square
        width = int(ceil(sqrt(n_tiles)))
        height = width
    else:
        assert isinstance(desired_width, int)
        width = desired_width
        height = int(ceil(float(n_tiles) / width))
    return height, width


def get_tiles_height_width_ratio(n_tiles, width_ratio=1.0):
    '''Get a height x width size that will fit n_tiles tiles.'''
    width = int(ceil(sqrt(n_tiles * width_ratio)))
    return get_tiles_height_width(n_tiles, desired_width=width)


def tile_images_normalize(data, c01=False, boost_indiv=0.0, boost_gamma=1.0, single_tile=False, scale_range=1.0,
                          neg_pos_colors=None):
    data = data.copy()
    if single_tile:
        # promote 2D image -> 3D batch (01 -> b01) or 3D image -> 4D batch (01c -> b01c OR c01 -> bc01)
        data = data[newaxis]
    if c01:
        # Convert bc01 -> b01c
        assert len(data.shape) == 4, 'expected bc01 data'
        data = data.transpose(0, 2, 3, 1)

    if neg_pos_colors:
        neg_clr, pos_clr = neg_pos_colors
        neg_clr = array(neg_clr).reshape((1, 3))
        pos_clr = array(pos_clr).reshape((1, 3))
        # Keep 0 at 0
        data /= max(data.max(), -data.min()) + 1e-10  # Map data to [-1, 1]

        # data += .5 * scale_range  # now in [0, scale_range]
        # assert data.min() >= 0
        # assert data.max() <= scale_range
        if len(data.shape) == 3:
            data = data.reshape(data.shape + (1,))
        assert data.shape[3] == 1, 'neg_pos_color only makes sense if color data is not provided (channels should be 1)'
        data = dot((data > 0) * data, pos_clr) + dot((data < 0) * -data, neg_clr)

    data -= data.min()
    data *= scale_range / (data.max() + 1e-10)

    # sqrt-scale (0->0, .1->.3, 1->1)
    assert boost_indiv >= 0 and boost_indiv <= 1, 'boost_indiv out of range'
    # print 'using boost_indiv:', boost_indiv
    if boost_indiv > 0:
        if len(data.shape) == 4:
            mm = (data.max(-1).max(-1).max(-1) + 1e-10) ** -boost_indiv
        else:
            mm = (data.max(-1).max(-1) + 1e-10) ** -boost_indiv
        data = (data.T * mm).T
    if boost_gamma != 1.0:
        data = data ** boost_gamma

    # Promote single-channel data to 3 channel color
    if len(data.shape) == 3:
        # b01 -> b01c
        data = tile(data[:, :, :, newaxis], 3)

    return data


def tile_images_make_tiles(data, padsize=1, padval=0, hw=None, highlights=None):
    if hw:
        height, width = hw
    else:
        height, width = get_tiles_height_width(data.shape[0])
    assert height * width >= data.shape[0], '{} rows x {} columns cannot fit {} tiles'.format(height, width,
                                                                                              data.shape[0])

    # First iteration: one-way padding, no highlights
    # padding = ((0, width*height - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    # data = pad(data, padding, mode='constant', constant_values=(padval, padval))

    # Second iteration: padding with highlights
    # padding = ((0, width*height - data.shape[0]), (padsize, padsize), (padsize, padsize)) + ((0, 0),) * (data.ndim - 3)
    # print 'tile_images: data min,max =', data.min(), data.max()
    # padder = SmartPadder()
    ##data = pad(data, padding, mode=jy_pad_fn)
    # data = pad(data, padding, mode=padder.pad_function)
    # print 'padder.calls =', padder.calls

    # Third iteration: two-way padding with highlights
    if highlights is not None:
        assert len(highlights) == data.shape[0]
    padding = ((0, width * height - data.shape[0]), (padsize, padsize), (padsize, padsize)) + ((0, 0),) * (
            data.ndim - 3)

    # First pad with constant vals
    try:
        len(padval)
    except:
        padval = tuple((padval,))
    assert len(padval) in (1, 3), 'padval should be grayscale (len 1) or color (len 3)'
    if len(padval) == 1:
        data = pad(data, padding, mode='constant', constant_values=(padval, padval))
    else:
        data = pad(data, padding, mode='constant', constant_values=(0, 0))
        for cc in (0, 1, 2):
            # Replace 0s with proper color in each channel
            data[:padding[0][0], :, :, cc] = padval[cc]
            if padding[0][1] > 0:
                data[-padding[0][1]:, :, :, cc] = padval[cc]
            data[:, :padding[1][0], :, cc] = padval[cc]
            if padding[1][1] > 0:
                data[:, -padding[1][1]:, :, cc] = padval[cc]
            data[:, :, :padding[2][0], cc] = padval[cc]
            if padding[2][1] > 0:
                data[:, :, -padding[2][1]:, cc] = padval[cc]
    if highlights is not None:
        # Then highlight if necessary
        for ii, highlight in enumerate(highlights):
            if highlight is not None:
                data[ii, :padding[1][0], :, :] = highlight
                if padding[1][1] > 0:
                    data[ii, -padding[1][1]:, :, :] = highlight
                data[ii, :, :padding[2][0], :] = highlight
                if padding[2][1] > 0:
                    data[ii, :, -padding[2][1]:, :] = highlight

    # tile the filters into an image
    data = data.reshape((height, width) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((height * data.shape[1], width * data.shape[3]) + data.shape[4:])
    data = data[0:-padsize, 0:-padsize]  # remove excess padding

    return (height, width), data


def to_255(vals_01):
    '''Convert vals in [0,1] to [0,255]'''
    try:
        ret = [v * 255 for v in vals_01]
        if type(vals_01) is tuple:
            return tuple(ret)
        else:
            return ret
    except TypeError:
        # Not iterable (single int or float)
        return vals_01 * 255


def ensure_uint255_and_resize_to_fit(img, out_max_shape,
                                     shrink_interpolation=cv2.INTER_LINEAR,
                                     grow_interpolation=cv2.INTER_NEAREST):
    as_uint255 = ensure_uint255(img)
    return resize_to_fit(as_uint255, out_max_shape,
                         dtype_out='uint8',
                         shrink_interpolation=shrink_interpolation,
                         grow_interpolation=grow_interpolation)


def ensure_uint255(arr):
    '''If data is float, multiply by 255 and convert to uint8. Else leave as uint8.'''
    if arr.dtype == 'uint8':
        return arr
    elif arr.dtype in ('float32', 'float64'):
        # print 'extra check...'
        # assert arr.max() <= 1.1
        return array(arr * 255, dtype='uint8')
    else:
        raise Exception('ensure_uint255 expects uint8 or float input but got %s with range [%g,%g,].' % (
            arr.dtype, arr.min(), arr.max()))


def ensure_float01(arr, dtype_preference='float32'):
    '''If data is uint, convert to float and divide by 255. Else leave at float.'''
    if arr.dtype == 'uint8':
        # print 'extra check...'
        # assert arr.max() <= 256
        return array(arr, dtype=dtype_preference) / 255
    elif arr.dtype in ('float32', 'float64'):
        return arr
    else:
        raise Exception('ensure_float01 expects uint8 or float input but got %s with range [%g,%g,].' % (
            arr.dtype, arr.min(), arr.max()))


def resize_to_fit(img, out_max_shape,
                  dtype_out=None,
                  shrink_interpolation=cv2.INTER_LINEAR,
                  grow_interpolation=cv2.INTER_NEAREST):
    '''Resizes to fit within out_max_shape. If ratio is different,
    returns an image that fits but is smaller along one of the two
    dimensions.

    If one of the out_max_shape dimensions is None, then use only the other dimension to perform resizing.

    Timing info on MBP Retina with OpenBlas:
     - conclusion: uint8 is always tied or faster. float64 is slower.

    Scaling down:
    In [79]: timeit.Timer('resize_to_fit(aa, (200,200))', setup='from kerasvis.app import resize_to_fit; import numpy as np; aa = array(np.random.uniform(0,255,(1000,1000,3)), dtype="uint8")').timeit(100)
    Out[79]: 0.04950380325317383

    In [77]: timeit.Timer('resize_to_fit(aa, (200,200))', setup='from kerasvis.app import resize_to_fit; import numpy as np; aa = array(np.random.uniform(0,255,(1000,1000,3)), dtype="float32")').timeit(100)
    Out[77]: 0.049156904220581055

    In [76]: timeit.Timer('resize_to_fit(aa, (200,200))', setup='from kerasvis.app import resize_to_fit; import numpy as np; aa = array(np.random.uniform(0,255,(1000,1000,3)), dtype="float64")').timeit(100)
    Out[76]: 0.11808204650878906

    Scaling up:
    In [68]: timeit.Timer('resize_to_fit(aa, (2000,2000))', setup='from kerasvis.app import resize_to_fit; import numpy as np; aa = array(np.random.uniform(0,255,(1000,1000,3)), dtype="uint8")').timeit(100)
    Out[68]: 0.4357950687408447

    In [70]: timeit.Timer('resize_to_fit(aa, (2000,2000))', setup='from kerasvis.app import resize_to_fit; import numpy as np; aa = array(np.random.uniform(0,255,(1000,1000,3)), dtype="float32")').timeit(100)
    Out[70]: 1.3411099910736084

    In [73]: timeit.Timer('resize_to_fit(aa, (2000,2000))', setup='from kerasvis.app import resize_to_fit; import numpy as np; aa = array(np.random.uniform(0,255,(1000,1000,3)), dtype="float64")').timeit(100)
    Out[73]: 2.6078310012817383
    '''

    if dtype_out is not None and img.dtype != dtype_out:
        dtype_in_size = img.dtype.itemsize
        dtype_out_size = dtype(dtype_out).itemsize
        convert_early = (dtype_out_size < dtype_in_size)
        convert_late = not convert_early
    else:
        convert_early = False
        convert_late = False

    if img.shape[0] == 0 and img.shape[1] == 0:
        scale = 1
    elif out_max_shape[0] is None or img.shape[0] == 0:
        scale = float(out_max_shape[1]) / img.shape[1]
    elif out_max_shape[1] is None or img.shape[1] == 0:
        scale = float(out_max_shape[0]) / img.shape[0]
    else:
        scale = min(float(out_max_shape[0]) / img.shape[0],
                    float(out_max_shape[1]) / img.shape[1])

    if convert_early:
        img = array(img, dtype=dtype_out)
    out = cv2.resize(img,
                     (int(img.shape[1] * scale), int(img.shape[0] * scale)),  # in (c,r) order
                     interpolation=grow_interpolation if scale > 1 else shrink_interpolation)

    if convert_late:
        out = array(out, dtype=dtype_out)
    return out


class FormattedString(object):
    def __init__(self, string, defaults, face=None, fsize=None, clr=None, thick=None, align=None, width=None):
        self.string = string
        self.face = face if face else defaults['face']
        self.fsize = fsize if fsize else defaults['fsize']
        self.clr = clr if clr else defaults['clr']
        self.thick = thick if thick else defaults['thick']
        self.width = width  # if None: calculate width automatically
        self.align = align if align else defaults.get('align', 'left')


def cv2_typeset_text(data, lines, loc, between=' ', string_spacing=0, line_spacing=0, wrap=False):
    '''Typesets mutliple strings on multiple lines of text, where each string may have its own formatting.

    Given:
    data: as in cv2.putText
    loc: as in cv2.putText
    lines: list of lists of FormattedString objects, may be modified by this function!
    between: what to insert between each string on each line, ala str.join
    string_spacing: extra spacing to insert between strings on a line
    line_spacing: extra spacing to insert between lines
    wrap: if true, wraps words to next line

    Returns:
    locy: new y location = loc[1] + y-offset resulting from lines of text
    '''

    data_width = data.shape[1]

    # lines_modified = False
    # lines = lines_in    # will be deepcopied if modification is needed later

    if isinstance(lines, FormattedString):
        lines = [lines]
    assert isinstance(lines,
                      list), 'lines must be a list of lines or list of FormattedString objects or a single FormattedString object'
    if len(lines) == 0:
        return loc[1]
    if not isinstance(lines[0], list):
        # If a single line of text is given as a list of strings, convert to multiline format
        lines = [lines]

    locy = loc[1]

    line_num = 0
    while line_num < len(lines):
        line = lines[line_num]
        maxy = 0
        locx = loc[0]
        for ii, fs in enumerate(line):
            last_on_line = (ii == len(line) - 1)
            if not last_on_line:
                fs.string += between
            boxsize, _ = cv2.getTextSize(fs.string, fs.face, fs.fsize, fs.thick)
            if fs.width is not None:
                if fs.align == 'right':
                    locx += fs.width - boxsize[0]
                elif fs.align == 'center':
                    locx += (fs.width - boxsize[0]) / 2
                    # print 'right boundary is', locx + boxsize[0], '(%s)' % fs.string
                    #                print 'HERE'
            right_edge = locx + boxsize[0]
            if wrap and ii > 0 and right_edge > data_width:
                # Wrap rest of line to the next line
                # if not lines_modified:
                #    lines = deepcopy(lines_in)
                #    lines_modified = True
                new_this_line = line[:ii]
                new_next_line = line[ii:]
                lines[line_num] = new_this_line
                lines.insert(line_num + 1, new_next_line)
                break
                ###line_num += 1
                ###continue
            cv2.putText(data, fs.string, (locx, locy), fs.face, fs.fsize, fs.clr, fs.thick)
            maxy = max(maxy, boxsize[1])
            if fs.width is not None:
                if fs.align == 'right':
                    locx += boxsize[0]
                elif fs.align == 'left':
                    locx += fs.width
                elif fs.align == 'center':
                    locx += fs.width - (fs.width - boxsize[0]) / 2
            else:
                locx += boxsize[0]
            locx += string_spacing
        line_num += 1
        locy += maxy + line_spacing

    return locy


def saveimage(filename, im):
    '''Saves an image with pixel values in [0,1]'''
    # matplotlib.image.imsave(filename, im)
    if len(im.shape) == 3:
        # Reverse RGB to OpenCV BGR order for color images
        cv2.imwrite(filename, 255 * im[:, :, ::-1])
    else:
        cv2.imwrite(filename, 255 * im)


def saveimagesc(filename, im):
    saveimage(filename, norm01(im))


def saveimagescc(filename, im, center):
    saveimage(filename, norm01c(im, center))

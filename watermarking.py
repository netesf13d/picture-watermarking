# -*- coding: utf-8 -*-
"""
Functions for watermarking pictures through the construction of masks
superimposed on the picture.
* <watermark_mask> : To cover the picture with lines of a repeated text.   
* <text_mask> : single text mask at given position on the picture.
* <tracking_dots_mask> : tracking dots mask encoding date and a short text.
* <decode_tracking_dots> : decode the tracking dots mask on a picture
* <mask_over> : superimpose a mask over a picture

These masks can be used for both regular watermarking as well as LSB
steganography.
"""

import datetime

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Functions
# =============================================================================

def _bbox_size(text: str, dpi: float, fontdict: dict)-> np.ndarray:
    """
    Get the size (sz_x, sz_y) of a text bbox in pixels.
    """
    fig = plt.figure(dpi=dpi)
    bbox = fig.text(0, 0, text, **fontdict).get_window_extent()._points
    plt.close(fig)
    return bbox[1] - bbox[0]


def watermark_mask(im_size: tuple[int, int],
                   dpi: float,
                   text: str,
                   text_join: str = '    ',
                   text_fontdict: dict | None = None,
                   shifts: tuple[float, float] = (50, 200),
                   offsets: tuple[float, float] = (0, 0),
                   angle: float = 30)-> np.ndarray:
    """
    Return an array representing an RGBA picture with a watermark text. The
    text is repeated both horizontally and vertically to cover the entire
    image.

    Parameters
    ----------
    im_size : tuple[int, int]
        Size (dx, dy) of the mask, in pixels. It should be the same as that of
        the original image.
    dpi : float
        Dots per inch.
    text : str
        Watermark text.
    text_join : str, optional
        String which joins repetitions of the watermark text.
        The default is '    '.
    text_fontdict : dict | None, optional
        Text font dict containing text parameters passed to Axes.text.
        The default is None.
    shifts : tuple[float, float], optional
        Shifts between successive lines, in pixels. The shifts apply in the
        frame where the lines are horizontal.
        The default is (50, 250).
    offsets : tuple[float, float] (x0, y0), optional
        Lines position offsets, in pixels.
        The default is (0, 0).
    angle : float, optional
        Rotation angle of the text, in degrees. Must be in [0, 90).
        The default is 30.

    Returns
    -------
    np.ndarray
        The mask as an array. Contains RGBA channels, shape im_size + (4,)

    """
    assert 0 <= angle < 90, "angle must be in [0, 90)"
    assert abs(shifts[0]) > 1, "lines must be shifted by at least 1 pixel"
    text_fontdict = {} if text_fontdict is None else text_fontdict
    
    ## build figure
    fig = plt.figure(figsize=[float(sz/dpi) for sz in im_size], dpi=dpi,
                     facecolor=(0, 0, 0, 0))
    ax = fig.add_axes((0, 0, 1, 1), facecolor=(0, 0, 0, 0))
    ax.set_axis_off()
    ax.set_xlim(-0.5, im_size[0]-0.5)
    ax.set_ylim(im_size[1]-0.5, -0.5)
    
    ## draw text
    bbox_sz = _bbox_size(text, dpi, text_fontdict)
    theta = angle * np.pi / 180
    
    dx = np.cos(theta) * shifts[0] + np.sin(theta) * shifts[1]
    dy = -np.sin(theta) * shifts[0] + np.cos(theta) * shifts[1]
    delta = shifts[1] / np.cos(theta)
    
    nlines = int((im_size[1] + im_size[0]*np.tan(theta)) / delta)
    repeat = int((im_size[0] + nlines*shifts[0]) / (np.cos(theta)*bbox_sz[0])) + 1
    txt = text_join.join((text,)*repeat)
    
    for i in range(nlines + 1):
        ax.text(offsets[0] + i*dx, offsets[1] + i*dy,
                txt, rotation=angle, ha='center', va='center', **text_fontdict)
    
    ## convert figure to numpy array
    fig.canvas.draw()
    mask = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig) # prevents the figure from being drawn
    return mask


def text_mask(im_size: tuple[int, int],
              dpi: float,
              text: str,
              text_pos: tuple[float, float],
              text_fontdict: dict | None = None)-> np.ndarray:
    """
    Return an array representing an RGBA picture with a single text at given
    position.

    Parameters
    ----------
    im_size : tuple[int, int]
        Size (dx, dy) of the mask, in pixels. It should be the same as that of
        the original image.
    dpi : float
        Dots per inch.
    text : str
        Watermark text.
    text_pos : tuple[float, float]
        The position of the text (bottom-left of the bbox), in pixels.
    text_fontdict : dict | None, optional
        Text font dict containing text parameters passed to Axes.text.
        The default is None.

    Returns
    -------
    np.ndarray
        The mask as an array. Contains RGBA channels, shape im_size + (4,)

    """
    text_fontdict = {} if text_fontdict is None else text_fontdict
    
    ## build figure
    fig = plt.figure(figsize=[float(sz/dpi) for sz in im_size], dpi=dpi,
                     facecolor='none')
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(-0.5, im_size[0]-0.5) # match plt.imshow xlim and ylim
    ax.set_ylim(im_size[1]-0.5, -0.5)
    
    ## draw text
    ax.text(*text_pos, text, ha='left', va='bottom', **text_fontdict)
    
    ## convert figure to numpy array
    fig.canvas.draw()
    mask = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig) # prevents the figure from being drawn
    return mask


def tracking_dots_mask(
        im_size: tuple[int, int],
        loc: tuple[int, int],
        date: datetime.date | None = None,
        text: str = '',
        dot_spacing: int = 4,
        dot_size: int = 1,
        rgba_color: tuple[int, int, int, int] = (0xff, 0xff, 0, 0x80)
        )-> np.ndarray:
    """
    Return an RGBA mask of dots encoding a date and a short text. This is
    analogous to tracking dots found in printers.
    https://en.wikipedia.org/wiki/Printer_tracking_dots
    
    The encoded data is:
    - date is encoded as an uint32, the date ordinal (nb of days since 1-01-01)
    - text is encoded in utf-8
    The dots are displayed as an array (4 + len(text), 8), data | text in big
    endian format. A colored dot encodes bit 1.

    Parameters
    ----------
    im_size : tuple[int, int]
        Size (dx, dy) of the mask, in pixels. It should be the same as that of
        the original image.
    loc : tuple[int, int] (x0, y0)
        Position of the top left point of the array.
    date : datetime.date | None, optional
        The date to be encoded.
        The default is None, which encodes today's date.
    text : str, optional
        The text. Its uft-8 encoded form must not be longer than 12 bytes.
        The default is ''.
    dot_spacing : int, optional
        Space between dots, in pixels. The default is 4.
    dot_size : int, optional
        The size of the dots, in pixels. The default is 1.
    rgba_color : tuple[int, int, int, int], optional
        The color of the dots in the mask.
        The default is (0xff, 0xff, 0, 0x80), yellow with alpha = 0.5.

    Returns
    -------
    mask : np.ndarray[np.uint8] of shape im_size + (4,)
        The tracking dots mask.

    """
    text = text.encode('utf-8')
    n = len(text)
    assert n <= 12, "utf-8 encoded text must have length <= 12"
    date = datetime.date.today() if date is None else date
    
    bits = f'{date.toordinal():032b}' + ''.join(f'{c:08b}' for c in text)
    bits = np.fromiter(bits, dtype=int)
    
    # indices of squares to mark
    if loc[0] == 0 or loc[1] == 0:
        raise ValueError("loc values cannot be 0")
    delta = dot_spacing + dot_size
    x1 = loc[0] + (len(bits) // 8) * delta
    if x1 > im_size[0] or (loc[0] < 0 and x1 >= 0):
        raise ValueError("out of range along x-coord")
    y1 = loc[1] + 8 * delta
    if y1 > im_size[1] or (loc[1] < 0 and y1 >= 0):
        raise ValueError("out of range along y-coord")
    idx_x = (loc[0] + (np.arange(len(bits)) // 8) * delta)[np.nonzero(bits)]
    idx_y = (loc[1] + (np.arange(len(bits)) % 8) * delta)[np.nonzero(bits)]

    mask = np.full((im_size[1], im_size[0], 4), 255, dtype=np.uint8)
    mask[:, :, 3] = 0
    for j, i in zip(idx_x, idx_y):
        mask[i:i+dot_size, j:j+dot_size] = rgba_color
    
    return mask


def decode_tracking_dots(im_arr: np.ndarray,
                       loc: tuple[int, int],
                       dot_spacing: int = 4,
                       dot_size: int = 1)-> tuple[datetime.date, str]:
    """
    Decode tracking dots embedded in a picture.
    
    The encoded data is date | text:
    - date is encoded as an uint32, the date ordinal (nb of days since 1-01-01)
    - text is encoded in utf-8

    Parameters
    ----------
    im_arr : np.ndarray
        Array representing the picture in RGBA.
    loc : tuple[int, int] (x0, y0)
        Position of the top left point of the array.
    dot_spacing : int, optional
        Space between dots, in pixels. The default is 4.
    dot_size : int, optional
        The size of the dots, in pixels. The default is 1.

    Returns
    -------
    date : datetime.date
        The encoded date.
    text : str
        The encoded text.

    """
    # indices of marked squares
    if loc[0] == 0 or loc[1] == 0:
        raise ValueError("loc values cannot be 0")
    delta = dot_spacing + dot_size
    if loc[0] >= 0:
        ncols = min((im_arr.shape[1] - loc[0]) // delta, 16)
    else:
        ncols = min((-loc[0]) // delta, 16)
    if ncols < 3:
        raise ValueError("out of range along x-coord")
    y1 = loc[1] + 8 * delta
    if y1 > im_arr.shape[0] or (loc[1] < 0 and y1 >= 0):
        raise ValueError("out of range along y-coord")
    idx_x = (loc[0] + (np.arange(8*ncols) // 8) * delta)
    idx_y = (loc[1] + (np.arange(8*ncols) % 8) * delta)
    
    # extract colors at dot locations
    dot_colors = []
    for j, i in zip(idx_x, idx_y):
        c = np.mean(im_arr[i:i+dot_size, j:j+dot_size], axis=(0, 1))
        dot_colors.append(c)
    dot_colors = np.array(dot_colors)
    # extract surrounding colors
    base_colors = []
    for j, i in zip(idx_x, idx_y):
        c = [im_arr[i-1, j:j+dot_size], im_arr[i+dot_size+1, j:j+dot_size],
             im_arr[i:i+dot_size, j-1], im_arr[i:i+dot_size, j+dot_size+1]]
        base_colors.append(np.mean(c, axis=(0, 1)))
    base_colors = np.array(base_colors)
    # color differences between dots and surroundings
    diff = (dot_colors - base_colors)[:, :3]
    # detect dots
    norm = np.sqrt(np.sum(diff**2, axis=1))
    thr = (np.max(norm) - np.min(norm)) / 2
    bytestr = bytes(np.packbits(norm > thr))
    # decode date
    date_ord = int.from_bytes(bytestr[:4], 'big')
    date = datetime.date.fromordinal(date_ord)
    # decode text
    text = bytestr[4:].strip(b'\x00').decode('utf-8')
    
    return date, text


def mask_over(im_arr: np.ndarray,
              mask: np.ndarray)-> np.ndarray:
    """
    Compose mask over an image.
    See https://en.wikipedia.org/wiki/Alpha_compositing

    Parameters
    ----------
    im_arr : np.ndarray
        The image RGBA array.
    mask : np.ndarray
        The mask RGBA array, composed over the image.

    """
    im_rgb = im_arr[:, :, :3] / 255
    mask_rgb = mask[:, :, :3] / 255
    im_alpha = im_arr[:, :, [3]] / 255
    mask_alpha = mask[:, :, [3]] / 255
    
    img = np.empty_like(im_arr, dtype=np.float32)
    img[:, :, -1] = (mask_alpha + im_alpha * (1-mask_alpha))[:, :, 0]
    img[:, :, :3] = mask_rgb * mask_alpha + im_rgb * im_alpha * (1-mask_alpha)
    img[:, :, :3] /= img[:, :, [3]]
    
    return np.round(img * 255).astype(np.uint8)


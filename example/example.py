# -*- coding: utf-8 -*-
"""
Script illustrating the use of watermarking functions.
"""

import datetime
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# module conversations in parent directory
wmdir = Path(__file__).resolve().parents[1]
if not str(wmdir) in sys.path:
    sys.path.append(str(wmdir))
from watermarking import (watermark_mask, text_mask,
                          tracking_dots_mask, decode_tracking_dots,
                          mask_over)

# =============================================================================
# Parameters
# =============================================================================

fn_in = "./Hopper_Navy-Portrait.jpg"
fn_out = "./watermark_Hopper_Navy-Portrait.png"

# watermark
watermark = "this is a watermark"
wm_join = '    ' # string joining text repetitions
wm_font = {'fontsize': 24, 'fontweight': 'bold',
           'color': '0.5', 'alpha': 0.5}
wm_shifts = (150, 120) # pixels
wm_offsets = (0, 0) # pixels
wm_angle = 30 # degrees, 0 <= wm_angle <= 90

# single text
text = "this is a single text"
txt_font = {'fontsize': 12, 'color': '0.9', 'alpha': 0.8}
txt_pos = (510, 760) # pixels

# stego watermark
steg = "this is hidden"
s_join = '    ' # string joining text repetitions
s_font = {'fontsize': 24, 'fontweight': 'bold', 'color': 'k', 'alpha': 1}
s_shifts = (50, 120) # pixels
s_offsets = (0, 60) # pixels
s_angle = 30 # degrees, 0 <= wm_angle <= 90

# yellow dots message
td_date = datetime.date(1992, 1, 1)
td_text = "yellow dots"
td_loc = (150, 450)
td_size = 1
td_spacing = 4
td_color = (0xff, 0xff, 0x0, 0x40)


# =============================================================================
# Script
# =============================================================================

##### Load picture to be watermarked #####
with Image.open(fn_in).convert('RGBA') as base:
    size = base.size
    dpi = base.info['dpi']
    im = np.array(base)
dpi = dpi[0]
im_sz = (float(size[0] / dpi), float(size[1] / dpi))

##### Make masks #####
# watermark mask
wm_mask = watermark_mask(size, dpi, watermark, wm_join, wm_font,
                          wm_shifts, wm_offsets, wm_angle)
# signle text mask
txt_mask = text_mask(size, dpi, text, txt_pos, txt_font)
# stego mask
steg_mask = watermark_mask(size, dpi, steg, s_join, s_font,
                            s_shifts, s_offsets, s_angle)
steg_mask = (steg_mask[:, :, 0] < 128).astype(np.uint8)[..., np.newaxis]
# yellow dots mask
td_mask = tracking_dots_mask(size, td_loc, date=td_date, text=td_text,
                             dot_size=td_size, dot_spacing=td_spacing,
                             rgba_color=td_color)

##### assemble image #####
img = mask_over(im, wm_mask)
img = mask_over(img, txt_mask)
img = mask_over(img, td_mask)
img[:, :, :3] = (img[:, :, :3] & 0xfe) + steg_mask
pic = Image.fromarray(img).convert('RGB')
pic.save(fn_out)
pic.show()

##### Recover yellow dots encoded data #####
with Image.open(fn_out).convert('RGBA') as base:
    im_ = np.array(base)
r_date, r_text = decode_tracking_dots(im_, td_loc, td_spacing, td_size)
print(f"recovery {r_date == td_date and r_text == td_text} :",
      r_date, r_text)

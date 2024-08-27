Functions for watermarking pictures through the construction of masks
superimposed on the picture.
* <watermark_mask> : To cover the picture with lines of a repeated text.   
* <text_mask> : single text mask at given position on the picture.
* <tracking_dots_mask> : tracking dots mask encoding date and a short text.
* <decode_tracking_dots> : decode the tracking dots mask on a picture
* <mask_over> : superimpose a mask over a picture

These masks can be used for both regular watermarking as well as LSB
steganography.


### Dependencies

* numpy

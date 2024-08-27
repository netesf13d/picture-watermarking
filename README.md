# picture-watermarking

This small module contains functions useful for watermarking picture with custom text. It imlements:
* watermarking : covering the picture with lines of a repeated text
* single text marking : adding a text on the picture
* imprinting and decoding tracking dots : encode some data in small dots added to the picture (similar to [https://en.wikipedia.org/wiki/Printer_tracking_dots](https://en.wikipedia.org/wiki/Printer_tracking_dots))


## Usage

After picture loading and conversion to a numpy array `im`, obtain the picture `size` and `dpi`.
<p align="center">
    <img src="https://github.com/netesf13d/picture-watermarking/blob/main/example/Hopper_Navy-Portrait.jpg" width="300" />
</p>


#### Watermark mask
```
fontdict = {'fontsize': 24, 'fontweight': 'bold', 'color': '0.5', 'alpha': 0.5}
mask = watermark_mask(size, dpi, 'this is a watermark', text_fontdict=fontdict, shifts=(150, 120))
img = mask_over(im, mask)
```
<p align="center">
    <img src="https://github.com/netesf13d/picture-watermarking/blob/main/example/watermark_Hopper_Navy-Portrait.png" width="300" />
</p>


#### LSB steganography watermarking
```
fontdict = {'fontsize': 24, 'fontweight': 'bold', 'color': 'k', 'alpha': 1}
mask = watermark_mask(size, dpi, 'this is a watermark', text_fontdict=fontdict,
                      shifts=(150, 120), offsets=(0, 60))
img[:, :, :3] = (img[:, :, :3] & 0xfe) + mask
```
<p align="center">
    <img src="https://github.com/netesf13d/picture-watermarking/blob/main/example/hidden.png" width="300" />
</p>


#### Single text mask
```
fontdict = {'fontsize': 24, 'fontweight': 'bold', 'color': '0.9', 'alpha': 1}
mask = watermark_mask(size, dpi, 'this is a a single text', (510, 760), text_fontdict=fontdict)
img = mask_over(img, mask)
```
<p align="center">
    <img src="https://github.com/netesf13d/picture-watermarking/blob/main/example/single text.png" width="300" />
</p>


#### Tracking dots mask
```
mask = tracking_dots_mask(size, (150, 450), date=datetime.date(1992, 1, 1),
                          text='yellow dots', rgba_color=(0xff, 0xff, 0x0, 0x40))
img = mask_over(img, mask)
```
<p align="center">
    <img src="https://github.com/netesf13d/picture-watermarking/blob/main/example/yellow dots.png" width="500" />
</p>


## Dependencies

- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [python-pillow](https://python-pillow.org/) (for loading/saving pictures)


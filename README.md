# UHDRes: Ultra-High-Definition Image Restoration via Dual-Domain Decoupled Spectral Modulation

We propose UHDRes, a UHD image restoration framework solely based on the frequency domain and large-kernel convolution. Our method achieves state-of-the-art performance while maintaining high computational efficiency.
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHDRes.png)


##Datasets

## Pretrained Models
We provide pretrained models for UHD-LL, UHD-Haze, UHD-Blur, and 4K-Rain13k.
|  Task   | Model  |
|  ----   | ----   |
|  UHD-LL | [model](https://github.com/Zhao0100/UHDRes/blob/main/UHDRes_pretrained/UHD-LL.pth)   |
| UHD-Haze| [model](https://github.com/Zhao0100/UHDRes/blob/main/UHDRes_pretrained/UHD-Haze.pth) |
| 8KDehaze-mini| [model](https://github.com/Zhao0100/UHDRes/blob/main/UHDRes_pretrained/8KDehaze-mini.pth) |
| UHD-Blur| [model](https://github.com/Zhao0100/UHDRes/blob/main/UHDRes_pretrained/UHD-Blur.pth) |
| 4K-Rain13k| [model](https://github.com/Zhao0100/UHDRes/blob/main/UHDRes_pretrained/4K-Rain13k.pth) |

## Training
```python
bash train.sh
```

## Testing
```python
bash train.sh
```

### 4K Image LLIE
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-LL_tab.png)
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-LL.png)
### 4K Image Dehazing
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Haze_tab.png)
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Haze.png)
### 8K Image Dehazing
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/8KDehaze-mini_tab.png)
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/8KDehaze-mini.png)
### 4K Image Deblurring
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Blur_tab.png)
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Blur.png)
### 4K Image Deraining
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/4K-Rain13k_tab.png)
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/4K-Rain13k.png)

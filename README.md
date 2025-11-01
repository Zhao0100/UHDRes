# UHDRes: Ultra-High-Definition Image Restoration via Dual-Domain Decoupled Spectral Modulation

We propose UHDRes, a UHD image restoration framework solely based on the frequency domain and large-kernel convolution. Our method achieves state-of-the-art performance while maintaining high computational efficiency.
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHDRes.png)


## Dependencies and Installation
* CUDA 11.8 (or later)
* PyTorch 2.0.0 (or later)
```python
cd UHDRes
pip install -r requirements.txt
```

## Datasets
[UHD-LL](https://drive.google.com/drive/folders/1IneTwBsSiSSVXGoXQ9_hE1cO2d4Fd4DN), [UHD-Haze](https://drive.google.com/drive/folders/1PVCPkhqU_voPVFZj3FzAtUkJnQnF9lSa), [8KDehaze-mini](https://huggingface.co/datasets/fengyanzi/8KDehaze_mini), [UHDBlur](https://drive.google.com/drive/folders/1O6JYkOELLhpEkirAnxUB2JGWMqgwVvmX), [4K-Rain13k](https://pan.baidu.com/s/1Kao-OjWNlgg2Jl0Jtl7e5Q?pwd=spfi)

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
bash test.sh
```


# Results
<details>
<summary><h2>4K Image LLIE</h2></summary>
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-LL_tab.png" width="400px">
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-LL.png">
</details>

<details>
<summary><h2>4K Image Dehazing</h2></summary>
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Haze_tab.png" width="400px">
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Haze.png">
</details>

<details>
<summary><h2>8K Image Dehazing</h2></summary>
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/8KDehaze-mini_tab.png" width="400px">
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/8KDehaze-mini.png">
</details>

<details>
<summary><h2>4K Image Deblurring</h2></summary>
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Blur_tab.png" width="400px">
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Blur.png">
</details>

<details>
<summary><h2>4K Image Deraining</h2></summary>
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/4K-Rain13k_tab.png" width="400px">
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/4K-Rain13k.png">
</details>

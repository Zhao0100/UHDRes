# 🚀 IEEE TCSVT 2026 | UHDRes: Ultra-High-Definition Image Restoration via Dual-Domain Decoupled Spectral Modulation


[[TCSVT2026]](https://ieeexplore.ieee.org/document/11579382) [[📄 arXiv 2511.05009]](https://arxiv.org/abs/2511.05009) [[💻 GitHub Project]](https://github.com/Zhao0100/UHDRes)      [[🍍知乎]](https://zhuanlan.zhihu.com/p/1971881443413189240)

> **July 2026 🔥 Publication!** UHDRes has been published in **IEEE Transactions on Circuits and Systems for Video Technology**.

> **Nov. 2025 🔥 New SOTA!** UHDRes achieves **state-of-the-art** performance across **five UHD restoration tasks (4K & 8K)**  
> with **only 0.40M parameters** — outperforming previous methods up to **+2.3dB PSNR** while using **10× fewer parameters**.

We propose UHDRes, a UHD image restoration framework solely based on the frequency domain and large-kernel convolution. Our method achieves state-of-the-art performance while maintaining high computational efficiency.
![](https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHDRes.png)


## Dependencies and Installation
* CUDA 11.8 (or later)
* PyTorch 2.0.0 (or later)
```python
# git clone this repository
git clone https://github.com/Zhao0100/UHDRes.git
cd UHDRes

# create new anaconda env
conda create -n uhdres python=3.9
source activate uhdres

# install PyTorch
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

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


## Qualitative Results
<div align="center">

### 4K Image LLIE
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-LL.png"><br><br>

### 4K Image Dehazing
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Haze.png"><br><br>

### 8K Image Dehazing
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/8KDehaze-mini.png"><br><br>

### 4K Image Deblurring
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Blur.png"><br><br>

### 4K Image Deraining
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/4K-Rain13k.png"><br><br>

</div>

## Quantatitive Results
<div align="center">

### 4K Image LLIE
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-LL_tab.png" width="400px"><br>

### 4K Image Dehazing
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Haze_tab.png" width="400px"><br>

### 8K Image Dehazing
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/8KDehaze-mini_tab.png" width="400px"><br>

### 4K Image Deblurring
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/UHD-Blur_tab.png" width="400px"><br>

### 4K Image Deraining
<img src="https://github.com/Zhao0100/UHDRes/blob/main/Figs/4K-Rain13k_tab.png" width="400px"><br>

</div>


# Citation
```
@ARTICLE{UHDRes,
      title={UHDRes: Ultra-High-Definition Image Restoration via Dual-Domain Decoupled Spectral Modulation}, 
      author={S. Zhao and W. Lu and B. Wang and T. Wang and K. Zhang and H. Zhao},
      year={2025},
      eprint={2511.05009},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2511.05009}, 
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Zhao0100/UHDRes&type=date&legend=top-left)](https://www.star-history.com/#Zhao0100/UHDRes&type=date&legend=top-left)

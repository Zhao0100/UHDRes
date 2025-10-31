import argparse
import cv2
import glob
import os
from tqdm import tqdm

from basicsr.utils.img_util import img2tensor, tensor2img, imwrite
from basicsr.archs.UHDRes_arch import UHDRes

import torch

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')


from comput_psnr_ssim import calculate_ssim as ssim_gray
from comput_psnr_ssim import calculate_psnr as psnr_gray

def equalize_hist_color(img):
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    eq_image = cv2.merge(eq_channels)
    return eq_image

def get_residue_structure_mean(tensor, r_dim=1):
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = (max_channel[0] - min_channel[0])
    mean = torch.mean(tensor, dim=r_dim, keepdim=True)
    device = mean.device
    res_channel = res_channel / torch.max(mean, torch.full(size=mean.size(), fill_value=0.000001).to(device))
    return res_channel
import torch.nn.functional as F
def check_image_size(x,window_size=128):
    _, _, h, w = x.size()
    mod_pad_h = (window_size  - h % (window_size)) % (
                window_size )
    mod_pad_w = (window_size  - w % (window_size)) % (
                window_size)
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
    return x

def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    # print(model)
    print("The number of parameters: {}".format(num_params))
def main():
    """Inference demo for FeMaSR
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='',
                        help='Input image or folder')
    parser.add_argument('-g', '--gt', type=str,
                        default='',
                        help='groundtruth image')
    parser.add_argument('-w', '--weight', type=str,
                        default='',
                        help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=1, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600,
                        help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enhance_weight_path = args.weight
    EnhanceNet = UHDRes().to(device)
    EnhanceNet.load_state_dict(torch.load(enhance_weight_path)['params'], strict=True)
    EnhanceNet.eval()
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    ssim_all = 0
    psnr_all = 0
    lpips_all = 0
    num_img = 0
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        gt_path = args.gt
        file_name = path.split('/')[-1]

        gt_img = cv2.imread(os.path.join(gt_path, file_name), cv2.IMREAD_UNCHANGED)
        print('image name', path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)
        b, c, h, w = img_tensor.size()
        print('b, c, h, w = img_tensor.size()', img_tensor.size())

        with torch.no_grad():
            import time
            t0 = time.time()
            output = EnhanceNet.test(img_tensor)
            t1 = time.time()
            print('time:', t1-t0)
        output = output
        output = output[:, :, :h, :w]
        output_img = tensor2img(output)
        gray = True
        ssim = ssim_gray(output_img, gt_img)
        psnr = psnr_gray(output_img, gt_img)
        lpips_value = lpips(2 * torch.clip(img2tensor(output_img).unsqueeze(0) / 255.0, 0, 1) - 1,
                            2 * img2tensor(gt_img).unsqueeze(0) / 255.0 - 1).data.cpu().numpy()
        ssim_all += ssim
        psnr_all += psnr
        lpips_all += lpips_value
        num_img += 1
        print('num_img', num_img)
        print('ssim', ssim)
        print('psnr', psnr)
        print('lpips_value', lpips_value)
        save_path = os.path.join(args.output, f'{img_name}')
        imwrite(output_img, save_path)

        pbar.update(1)
    pbar.close()
    print('avg_ssim:%f' % (ssim_all / num_img))
    print('avg_psnr:%f' % (psnr_all / num_img))
    print('avg_lpips:%f' % (lpips_all / num_img))


if __name__ == '__main__':
    main()

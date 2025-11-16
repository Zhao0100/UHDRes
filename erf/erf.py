import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from functools import partial
from typing import Callable
import seaborn
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from utils import rgb2ycbcr
import torch
import torch.nn as nn
from torch import optim as optim
from torchvision import datasets, transforms
from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from basicsr.utils.options import dict2str, parse_options

# --- Keep the rest of the imports and setup as in your original code ---
root_path = './UHDRes_test.yml'
opt, _ = parse_options(root_path, is_train=False)
opt=opt['datasets']['UHDHaze']

class PairedImageDataset(data.Dataset):
    # --- Keep the dataset class as in your original code ---
    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task'] if 'task' in opt else None
        self.noise = opt['noise'] if 'noise' in opt else 0

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl,
                                                  self.task)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;

        if self.task == 'CAR':
            # image range: [0, 255], int., H W 1
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=False)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='grayscale', float32=False)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

        elif self.task == 'denoising_gray':  # Matlab + OpenCV version
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise / 255., img_gt.shape)
            img_gt = np.expand_dims(img_gt, axis=2)
            img_lq = np.expand_dims(img_lq, axis=2)

        elif self.task == 'denoising_color':
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise / 255., img_gt.shape)

        else:
            # image range: [0, 1], float32., H W 3
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)



if True:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    import seaborn as sns

    #   Set figure parameters
    large = 24;
    med = 24;
    small = 24
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (16, 10), # Retained original larger size
              'axes.labelsize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("white")
    # plt.rc('font', **{'family': 'Times New Roman'})
    plt.rcParams['axes.unicode_minus'] = False


# --- Modified analyze_erf function to handle negative values while mimicking original mapping ---
# ALGRITHOM will be applied after shifting the data to be non-negative.
def analyze_erf(source, dest="heatmap.png", ALGRITHOM=lambda x: np.power(x, 0.25)): # ALGRITHOM now expects non-negative input
    def heatmap(data, camp='RdYlGn', figsize=(16, 10), ax=None, save_path=None, center=None, cbar=False): # Restore original heatmap defaults
        plt.figure(figsize=figsize, dpi=40)
        ax = sns.heatmap(data,
                         xticklabels=False,
                         yticklabels=False, cmap=camp,
                         center=center, annot=False, ax=ax, cbar=cbar, annot_kws={"size": 24}, fmt='.2f')
        plt.savefig(save_path)
        plt.close() # Close the figure to free memory


    def analyze_erf_internal(data, args):
        print("Data min (raw):", np.min(data))
        print("Data max (raw):", np.max(data))

        # --- Shift data to make it non-negative before applying ALGRITHOM ---
        # Find the minimum value. If it's negative, we need to shift by its absolute value.
        min_val = np.min(data)
        shift_amount = max(0, -min_val) # Shift only if minimum is negative

        shifted_data = data + shift_amount
        print("Shifted data min:", np.min(shifted_data))
        print("Shifted data max:", np.max(shifted_data))

        epsilon = 1e-8 # Small epsilon to avoid power(0, ...) issues

        transformed_data = np.zeros_like(shifted_data)
        # Only apply power to values greater than epsilon to avoid np.power(0, 0.25) issues
        # and numerical instability near zero.
        positive_mask = shifted_data > epsilon
        transformed_data[positive_mask] = args.ALGRITHOM(shifted_data[positive_mask])

        print("Transformed data min:", np.min(transformed_data))
        print("Transformed data max:", np.max(transformed_data))

        # --- Normalize the transformed data to [0, 1] ---
        # Based on the original RepLKNet code's normalization logic.
        max_transformed_data = np.max(transformed_data)
        if max_transformed_data > epsilon:
             normalized_data = transformed_data / max_transformed_data
        else:
             normalized_data = np.zeros_like(transformed_data)


        print("Normalized data min:", np.min(normalized_data))
        print("Normalized data max:", np.max(normalized_data))

        # Use the original heatmap function and colormap
        heatmap(normalized_data, save_path=args.heatmap_save, camp='RdYlGn', center=None, cbar=False)
        print('heatmap saved at ', args.heatmap_save)

    class Args():
        ...

    args = Args()
    args.source = source
    args.heatmap_save = dest
    args.ALGRITHOM = ALGRITHOM # ALGRITHOM is now power(x, 0.25)

    os.makedirs(os.path.dirname(args.heatmap_save), exist_ok=True)
    analyze_erf_internal(args.source, args)


# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
# This function remains unchanged as it correctly calculates the raw gradients without ReLU.
def visualize_erf(MODEL: nn.Module = None, num_images=231, data_path="path_UHDdehaze",
                  save_path=f"./tmp/{time.time()}/erf.npy"):
    def get_input_grad(model, samples):
        outputs = model(samples)
        out_size = outputs.size()

        # Check if output size is valid before accessing center
        if out_size[2] == 0 or out_size[3] == 0:
             print(f"Warning: Output size is zero: {out_size}. Cannot calculate gradient.")
             return None

        # Choose the central output pixel. Ensure it's within bounds.
        center_y = out_size[2] // 2
        center_x = out_size[3] // 2
        # Boundary check
        if center_y >= out_size[2] or center_x >= out_size[3]:
            print(f"Warning: Calculated center pixel ({center_y}, {center_x}) out of bounds for output size {out_size}. Using closest valid pixel.")
            center_y = min(center_y, out_size[2] - 1)
            center_x = min(center_x, out_size[3] - 1)


        central_point = outputs[:, :, center_y, center_x].sum()

        # Check if central_point is a scalar tensor before calculating gradient
        if central_point.numel() == 1:
            grad = torch.autograd.grad(central_point, samples)
            grad = grad[0]
            # *** REMOVED: torch.nn.functional.relu(grad) ***
            # Keep the original gradient which includes negative values

            aggregated = grad.sum((0, 1))
            grad_map = aggregated.cpu().numpy()
            return grad_map
        else:
            print(f"Warning: Central point is not a scalar ({central_point.shape}). Skipping gradient calculation.")
            return None

    def main(args, MODEL: nn.Module = None):
        print("reading from datapath", args.data_path)
        # Use the globally available opt for dataset
        dataset = PairedImageDataset(opt)

        # Use batch_size=1 for ERF calculation
        test_loader = data.DataLoader(dataset,batch_size=1,shuffle=False)

        model = MODEL
        model.cuda().eval()

        meter = AverageMeter()
        # optimizer.zero_grad() # Not needed

        print(f"Processing up to {args.num_images} images from dataset...")
        processed_images_count = 0
        for idx,data_sample in enumerate(test_loader):
            if processed_images_count >= args.num_images:
                break

            # Resize input images to a consistent size for meaningful ERF comparison.
            input_erf_size = (160, 160)
            try:
                samples = F.interpolate(data_sample['lq'],size=input_erf_size, mode='bicubic', align_corners=False)
            except Exception as e:
                 print(f"Error during interpolation for image {idx}: {e}. Skipping.")
                 continue


            samples = samples.cuda(non_blocking=True)
            samples.requires_grad = True

            # optimizer.zero_grad() # Not needed

            with torch.enable_grad(): # Ensure gradients are computed
                contribution_scores = get_input_grad(model, samples)

            torch.cuda.empty_cache()

            if contribution_scores is not None:
                if np.isnan(np.sum(contribution_scores)):
                    print(f'Image {idx}: got NAN in gradient, skipping.')
                else:
                    print(f'Accumulating gradient for image {idx}. Meter count: {meter.count}')
                    meter.update(contribution_scores)
                    processed_images_count += 1
            else:
                 print(f'Image {idx}: skipping due to gradient calculation issue.')


        if meter.count > 0:
            print(f"Finished processing {meter.count} images.")
            return meter.avg
        else:
            print("No valid gradients were accumulated.")
            return None


    class Args():
        ...

    args = Args()
    args.num_images = num_images
    args.data_path = data_path
    args.save_path = save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    avg_grad_map = main(args, MODEL)

    # Save the accumulated gradient map
    if avg_grad_map is not None:
        np.save(save_path, avg_grad_map)
        print(f"Accumulated gradient map saved to {save_path}")
    else:
        print("No accumulated gradient map to save.")

    return avg_grad_map # Return the accumulated map


from uhdres import buildUHDRes

if __name__ == '__main__':
    showpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "show/erf")
    os.makedirs(showpath, exist_ok=True)

    init_model = buildUHDRes()

    ckpt_path = 'pretrained_model'

    try:
        checkpoint = torch.load(ckpt_path)
        if 'params' in checkpoint:
            init_model.load_state_dict(checkpoint['params'], strict=True)
            print(f"Loaded model weights from '{ckpt_path}' using 'params' key.")
        else:
            init_model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded model weights from '{ckpt_path}'.")
    except Exception as e:
        print(f"Error loading model weights from {ckpt_path}: {e}")
        exit()

    erf_data_path = opt['dataroot_lq']

    grad_map = visualize_erf(
        init_model,
        num_images=231,
        data_path=erf_data_path,
        save_path=f"./tmp/{time.time()}/erf_raw_grad_160x160.npy"
    )

    # Visualize the accumulated gradient map
    if grad_map is not None:
        analyze_erf(source=grad_map, dest=f"{showpath}/erf_UHDRes_160_shifted_power.png")
    else:
        print("ERF visualization skipped due to no accumulated gradient map.")


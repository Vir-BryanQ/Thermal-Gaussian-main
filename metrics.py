#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from datetime import datetime

def preallocate_vmem(vram=38):
    required_elements = int(vram * 1024 * 1024 * 1024 / 4)
    while True:
        try:
            occupied = torch.empty(required_elements , dtype=torch.float32, device='cuda')
            del occupied
            break
        except RuntimeError as e:
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            if 'CUDA out of memory' in str(e):
                free_memory, total_memory = torch.cuda.mem_get_info(torch.cuda.current_device())
                print(f"*** CUDA OOM: {free_memory / (1024 * 1024)}MiB is free [{t}]")
            else:
                print(f"### {e} [{t}]")

preallocate_vmem()

def readImages(color_renders_dir, color_gt_dir, thermal_renders_dir, thermal_gt_dir, ns_metric):
    color_renders = []
    color_gts = []
    thermal_renders = []
    thermal_gts = []
    image_names = []
    for fname in os.listdir(color_renders_dir):
        color_render = Image.open(color_renders_dir / fname)
        color_gt = Image.open(color_gt_dir / fname)
        thermal_render = Image.open(thermal_renders_dir / fname)
        thermal_gt = Image.open(thermal_gt_dir / fname)
        if ns_metric:
            color_renders.append(tf.to_tensor(color_render).unsqueeze(0)[:, :3, :, :])
            color_gts.append(tf.to_tensor(color_gt).unsqueeze(0)[:, :3, :, :])
            thermal_renders.append(tf.to_tensor(thermal_render).unsqueeze(0)[:, :3, :, :])
            thermal_gts.append(tf.to_tensor(thermal_gt).unsqueeze(0)[:, :3, :, :])
        else:
            color_renders.append(tf.to_tensor(color_render).unsqueeze(0)[:, :3, :, :].cuda())
            color_gts.append(tf.to_tensor(color_gt).unsqueeze(0)[:, :3, :, :].cuda())
            thermal_renders.append(tf.to_tensor(thermal_render).unsqueeze(0)[:, :3, :, :].cuda())
            thermal_gts.append(tf.to_tensor(thermal_gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return color_renders, color_gts, thermal_renders, thermal_gts, image_names

def evaluate(model_paths, args):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    psnr1 = PeakSignalNoiseRatio(data_range=1.0)
    ssim1 = SSIM(data_range=1.0, size_average=True, channel=3)
    lpips1 = LearnedPerceptualImagePatchSimilarity(normalize=True)

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                color_gt_dir = method_dir/ "gt_color"
                color_renders_dir = method_dir / "renders_color"
                thermal_gt_dir = method_dir/ "gt_thermal"
                thermal_renders_dir = method_dir / "renders_thermal"
                color_renders, color_gts, thermal_renders, thermal_gts, image_names = readImages(color_renders_dir, color_gt_dir, thermal_renders_dir, thermal_gt_dir, args.ns_metric)

                color_ssims = []
                color_psnrs = []
                color_lpipss = []
                thermal_ssims = []
                thermal_psnrs = []
                thermal_lpipss = []

                for idx in tqdm(range(len(color_renders)), desc="Metric evaluation progress"):
                    if args.ns_metric:
                        color_ssims.append(float(ssim1(color_renders[idx], color_gts[idx]).item()))
                        color_psnrs.append(float(psnr1(color_renders[idx], color_gts[idx])))
                        color_lpipss.append(float(lpips1(color_renders[idx], color_gts[idx])))
                        thermal_ssims.append(float(ssim1(thermal_renders[idx], thermal_gts[idx]).item()))
                        thermal_psnrs.append(float(psnr1(thermal_renders[idx], thermal_gts[idx])))
                        thermal_lpipss.append(float(lpips1(thermal_renders[idx], thermal_gts[idx])))
                    else:
                        color_ssims.append(ssim(color_renders[idx], color_gts[idx]))
                        color_psnrs.append(psnr(color_renders[idx], color_gts[idx]))
                        color_lpipss.append(lpips(color_renders[idx], color_gts[idx], net_type='vgg'))
                        thermal_ssims.append(ssim(thermal_renders[idx], thermal_gts[idx]))
                        thermal_psnrs.append(psnr(thermal_renders[idx], thermal_gts[idx]))
                        thermal_lpipss.append(lpips(thermal_renders[idx], thermal_gts[idx], net_type='vgg'))

                print("  color_SSIM : {:>12.7f}".format(torch.tensor(color_ssims).mean(), ".5"))
                print("  color_PSNR : {:>12.7f}".format(torch.tensor(color_psnrs).mean(), ".5"))
                print("  color_LPIPS: {:>12.7f}".format(torch.tensor(color_lpipss).mean(), ".5"))
                
                print("  thermal_SSIM : {:>12.7f}".format(torch.tensor(thermal_ssims).mean(), ".5"))
                print("  thermal_PSNR : {:>12.7f}".format(torch.tensor(thermal_psnrs).mean(), ".5"))
                print("  thermal_LPIPS: {:>12.7f}".format(torch.tensor(thermal_lpipss).mean(), ".5"))

                print("")

                full_dict[scene_dir][method].update({"color_SSIM": torch.tensor(color_ssims).mean().item(),
                                                        "color_PSNR": torch.tensor(color_psnrs).mean().item(),
                                                        "color_LPIPS": torch.tensor(color_lpipss).mean().item(),
                                                        "thermal_SSIM": torch.tensor(thermal_ssims).mean().item(),
                                                        "thermal_PSNR": torch.tensor(thermal_psnrs).mean().item(),
                                                        "thermal_LPIPS": torch.tensor(thermal_lpipss).mean().item()
                                                        })
                
                per_view_dict[scene_dir][method].update({"color_SSIM": {name: ssim for ssim, name in zip(torch.tensor(color_ssims).tolist(), image_names)},
                                                            "color_PSNR": {name: psnr for psnr, name in zip(torch.tensor(color_psnrs).tolist(), image_names)},
                                                            "color_LPIPS": {name: lp for lp, name in zip(torch.tensor(color_lpipss).tolist(), image_names)},
                                                            "thermal_SSIM": {name: ssim for ssim, name in zip(torch.tensor(thermal_ssims).tolist(), image_names)},
                                                            "thermal_PSNR": {name: psnr for psnr, name in zip(torch.tensor(thermal_psnrs).tolist(), image_names)},
                                                            "thermal_LPIPS": {name: lp for lp, name in zip(torch.tensor(thermal_lpipss).tolist(), image_names)}}
                                                            )

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(e)
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--ns_metric", action="store_true")
    args = parser.parse_args()
    evaluate(args.model_paths, args)

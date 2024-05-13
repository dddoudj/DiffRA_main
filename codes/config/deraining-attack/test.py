import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils
import cv2

import numpy as np
import torch
import torchvision
from IPython import embed
import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
           and "pretrain_model" not in key
           and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"],
                 eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)
lpips_fn = lpips.LPIPS(net='alex').to(device)

scale = opt['degradation']['scale']

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["lpips"] = []
    test_times = []
    sum = 0
    numsum0 = 0

    psnr_sum = 0
    ssim_sum = 0
    lp_score_sum = 0

    for i, test_data in enumerate(test_loader):
        single_img_psnr = []
        single_img_ssim = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        sum += 1

        #### input dataset_LQ
        LQ, GT = test_data["LQ"], test_data["GT"]
        noisy_state = sde.noise_state(LQ)

        model.feed_data(noisy_state, LQ, GT)
        tic = time.time()
        model.test(sde, save_states=True)
        toc = time.time()
        test_times.append(toc - tic)

        num0 = model.attacksuccess()
        numsum0 += num0
        avgasr0 = numsum0 / sum * 100
        print("the number of attack successful1 : %.6f  " % (
            numsum0))

        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        output = util.tensor2img(SR_img.squeeze())  # uint8
        LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8

        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")
        util.save_img(output, save_img_path)

        if need_GT:
            gt_img = GT_ / 255.0
            sr_img = output / 255.0

            crop_border = opt["crop_border"] if opt["crop_border"] else scale
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[
                                 crop_border:-crop_border, crop_border:-crop_border
                                 ]
                cropped_gt_img = gt_img[
                                 crop_border:-crop_border, crop_border:-crop_border
                                 ]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
            lp_score = lpips_fn(
                GT.to(device) * 2 - 1, SR_img.to(device) * 2 - 1).squeeze().item()
            psnr_sum += psnr
            ssim_sum += ssim
            lp_score_sum += lp_score

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            test_results["lpips"].append(lp_score)

            logger.info(
                "img{:3d}:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f}.".format(
                    i, img_name, psnr, ssim, lp_score
                )
            )

            print(
                "---------------------------------------------------------------------------------------------------------------")

    ave_lpips = lp_score_sum / sum
    ave_psnr = psnr_sum / sum
    ave_ssim = ssim_sum / sum
    logger.info(
        "----Average PSNR/SSIM/LPIPS ----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f}\n".format(
            ave_psnr, ave_ssim, ave_lpips
        )
    )

    logger.info(
        "----ASR\t: {:.6f}%; \n".format(
            avgasr0)
    )

    print(f"average test time: {np.mean(test_times):.4f}")
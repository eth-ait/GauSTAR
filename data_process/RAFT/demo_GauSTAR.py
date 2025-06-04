import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from tqdm import tqdm
from pathlib import Path

DEVICE = 'cuda'

# target_size = (1502, 2046)
target_size = None
scalar = 0.5

# pad = (100, 20, 50, 50)  # top, bottom, left, right
pad = None  # top, bottom, left, right


def load_image(imfile, target_size=None, scalar=None, pad=None):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., 0:3]
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    elif scalar:
        img = cv2.resize(img, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_AREA)

    if pad:
        img = img[pad[0]:-pad[1], pad[2]:-pad[3]]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, save_path):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # if pad:
    #     img = img[pad:-pad, pad:-pad]
    #     flo = flo[pad:-pad, pad:-pad]

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imwrite(save_path, img_flo[:, :, [2, 1, 0]])


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    args.path = Path(args.path)

    frame_0 = int(args.frame_0)
    frame_end = int(args.frame_end)
    interval = int(args.interval)
    cmr = dict(np.load(args.path / "rgb_cameras.npz"))
    cmr_num = cmr["shape"].shape[0]

    with torch.no_grad():
        for f_idx in range(frame_0, frame_end - interval, interval):
            if interval == 1:
                frame_save_dir = args.path / f"{f_idx:04d}/flow_bi/"
            elif interval == 2:
                frame_save_dir = args.path / f"{f_idx:04d}/flow_bi_2f/"
            else:
                raise RuntimeError("Interval Error!")

            os.makedirs(frame_save_dir, exist_ok=True)
            if pad:
                np.savetxt(frame_save_dir / "pad.txt", np.array(pad))
            print("Save flow at: ", frame_save_dir)

            for cmr_idx in tqdm(range(cmr_num)):
                image1 = load_image(args.path / f"{f_idx:04d}/images/img_{cmr_idx:04d}.jpg",
                                    target_size=target_size, scalar=scalar, pad=pad)
                image2 = load_image(args.path / f"{(f_idx + interval):04d}/images/img_{cmr_idx:04d}.jpg",
                                    target_size=target_size, scalar=scalar, pad=pad)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                _, f_flow_up = model(image1, image2, iters=20, test_mode=True)
                viz(image1, f_flow_up, str(frame_save_dir / f"{cmr_idx:04d}_f.jpg"))
                f_flow_unpad = padder.unpad(f_flow_up[0]).permute(1, 2, 0)
                # tf.imwrite(frame_save_dir + f"{cmr_idx:04d}_f.tiff", f_flow_unpad.cpu().numpy(), compression='zlib')
                np.savez_compressed(frame_save_dir / f"{cmr_idx:04d}_f.npz", flow=f_flow_unpad.cpu().numpy())

                _, b_flow_up = model(image2, image1, iters=20, test_mode=True)
                viz(image2, b_flow_up, str(frame_save_dir / f"{cmr_idx:04d}_b.jpg"))
                b_flow_unpad = padder.unpad(b_flow_up[0]).permute(1, 2, 0)
                # tf.imwrite(frame_save_dir + f"{cmr_idx:04d}_b.tiff", b_flow_unpad.cpu().numpy(), compression='zlib')
                np.savez_compressed(frame_save_dir / f"{cmr_idx:04d}_b.npz", flow=b_flow_unpad.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument("--frame_0", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=0)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)

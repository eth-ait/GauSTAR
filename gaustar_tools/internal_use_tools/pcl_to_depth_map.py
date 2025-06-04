import os
import json
import argparse
import struct
import cv2
import pathlib
import numpy as np
import open3d as o3d
from tqdm import tqdm
from xml.etree import ElementTree as ET

import tifffile as tf

def load_camera_matrices(calib_file, rescale_factor, mm_scale=True):
    """
    Loads camera intrinsics, extrinsics and distortion from parsed xml tree.

    Outputs a dictionary with keys:
    - camid
    - resolution
    - intrinsics
    - extrinsics
    - distortion
    """

    # Load extrinsics and intrinsics from the copied file.
    print("Loading calibration", calib_file)
    tree = ET.parse(calib_file).getroot()
    cameras = [[cam for cam in pod] for pod in tree]
    cameras = sum(cameras, [])

    rgb_cameras = list(
        set(filter(lambda x: "RGBCamera" in x.attrib["id"], cameras))
    )  # unique cam ids to avoid redundancy

    extrinsics = []
    intrinsics = []
    dist_coeff = []
    cam_ids = []
    res = []
    for cam in rgb_cameras:
        if cam.attrib["camid"] in cam_ids:
            continue
        cam_ids.append(cam.attrib["camid"])
        res.append(
            (
                int(cam.attrib["width"]) / float(rescale_factor),
                int(cam.attrib["height"]) / float(rescale_factor),
            )
        )
        for el in cam:
            if "Extrinsic" in el.tag:
                text = el.text.replace("\n", ",")[1:]
                ext_c = np.fromstring(text, sep=",").reshape((3, 4))
                if mm_scale:
                    ext_c[:3, -1] /= 1000
                extrinsics.append(ext_c)
            if "Intrinsic" in el.tag:
                text = el.text.replace("\n", ",")[1:]
                int_c = np.fromstring(text, sep=",").reshape((3, 3))
                int_c[:2, :] /= float(rescale_factor)
                intrinsics.append(int_c)
            if "Distortion" in el.tag:
                text = el.text.replace("\n", ",")[1:]
                dist_c = np.fromstring(text, sep=",")
                dist_coeff.append(dist_c)

    cameras_dict = {
        "camid": cam_ids,
        "resolution": res,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "distortion": dist_coeff,
    }

    return cameras_dict


def read_ply(input_file, format_in="ffffff"):
    input_f = open(input_file, "rb")

    # Read the header
    header = []
    line = ""
    while "end_header" not in line:
        line = input_f.readline().decode("utf-8").strip()
        header.append(line)

    element_size_in = struct.calcsize(format_in)

    num_elements = int(header[2].split()[2])
    points = []
    normals = []
    weights = []
    for _ in range(num_elements):
        data = input_f.read(element_size_in)
        x, y, z, nx, ny, nz = struct.unpack(format_in, data)
        points.append((x, y, z))
        normals.append((nx, ny, nz))
        # weights.append((w0, w1))

    return header, np.array(points), np.array(normals), np.array(weights)


def pcl_to_depth(
    pcl,
    width,
    height,
    intrinsic,
    extrinsic,
    depth_max=10.0,
    depth_scale=1.0,
    blur=False,
):
    extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
    pcd = o3d.t.geometry.PointCloud(pcl.astype(np.float32))
    pcd.point.positions /= 1000
    depth_map = pcd.project_to_depth_image(
        width,
        height,
        intrinsic,
        extrinsic,
        depth_scale=depth_scale,
        depth_max=depth_max,
    )

    depth_map_np = np.asarray(depth_map.to_legacy())

    if blur:
        depth_map_np = cv2.medianBlur(depth_map_np, 5)

    depth_map_np = np.clip(depth_map_np, 0, 60000)  # * 1000

    return depth_map_np


def main(args):
    width, height = (4092, 3004)
    rescale_factor = 4  # depth too sparse otherwise
    os.makedirs(args.out_dir, exist_ok=True)
    pods_dict = json.load(open("./pods.json", "r"))
    towers_dict = json.load(open("./towers.json", "r"))
    cameras = load_camera_matrices(
        calib_file=args.calib_file, rescale_factor=rescale_factor
    )
    intrinsics = cameras["intrinsics"]
    extrinsics = cameras["extrinsics"]
    pointcloud_files = pathlib.Path(args.source_dir).glob("*.ply")
    for idx_frame, pcl_f in enumerate(tqdm(pointcloud_files)):
        if idx_frame >= 10:
            break
        header, verts, normals, weights = read_ply(input_file=pcl_f)
        for tower, pods in pods_dict.items():
            # idx_full = np.unique(
            #     np.concatenate([np.where(weights[:, 0] == i)[0] for i in pods])
            # )
            verts_vis = verts #[idx_full]

            for cam_idx, cam_id in enumerate(tqdm(cameras["camid"])):
                if int(cam_id) not in towers_dict[tower]:
                    continue
                cam_dir = f"{args.out_dir}/dpt/{cam_id}"
                os.makedirs(cam_dir, exist_ok=True)
                cam_int = intrinsics[cam_idx]
                cam_ext = extrinsics[cam_idx]
                dpt = pcl_to_depth(
                    width=width // rescale_factor,
                    height=height // rescale_factor,
                    intrinsic=cam_int,
                    extrinsic=cam_ext,
                    pcl=verts_vis,
                    blur=True,  # apply gaussian blur to fill gaps
                )
                dpt = dpt.T[:, ::-1]

                # ZCW visual
                bg_idx = dpt == 0
                dpt[bg_idx] = 999

                dpt = cv2.resize(dpt, (height, width), cv2.INTER_LINEAR)
                tf.imwrite(f"{cam_dir}/{(idx_frame+1):06d}.tiff", dpt, compression='zlib')

                render_depth = np.minimum(10, np.array(dpt))
                render_depth = np.uint8(render_depth / 10 * 255.0)
                render_depth = cv2.applyColorMap(render_depth, cv2.COLORMAP_JET)
                cv2.imwrite(f"{cam_dir}/{(idx_frame+1):06d}.jpg", render_depth)

                # dpt = cv2.resize(dpt, (width, height)).astype(np.uint16)
                # cv2.imwrite(f"{cam_dir}/depth-f{(idx_frame+1):05d}.png", dpt)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_dir",
        default="",
        help="Source directory of mergedpoints *.ply files from stage",
    )

    parser.add_argument(
        "--calib_file",
        default="",
        help="Camera calibration xml file",
    )

    parser.add_argument(
        "--out_dir",
        default="tmp",
        help="Output directory of visible pointcloud",
    )

    main(parser.parse_args())

import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2

from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import VariableTopologyMeshes
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.material import Material


def render_mesh_depth_w_aitviewer(camera_path, mesh_folder, gstar_folder,
                                  test=False, wo_cxcy=False, from_humanrf=False, mesh_res="100k",
                                  frame_0=0, frame_end=0, interval=1,):
    """rendering 3d scan into different camera views,
    input:
        camera_path: the path to camera info
        mesh_folder: the path to the scan folder
        mask_save_folder: where to save the scan mask
        render_info_save_folder: where to save the pytorch3d rendering info:
            {row_idx: mask to select out vertices, N_V: #scan vertices}
    return:
        None
    """
    # init cameras
    camera_infos = dict(np.load(camera_path))
    shape = np.array(camera_infos["shape"], dtype=np.int32)
    cols = np.max(shape[:, 1])
    rows = np.max(shape[:, 0])

    viewer = HeadlessRenderer(size=(cols, rows))
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.shadows_enabled = False
    viewer.scene.lights = []


    if test:
        test_cmr_ids = [5, 15, 25, 35, 45]
        camera_infos["ids"] = camera_infos["ids"][test_cmr_ids]
        camera_infos["intrinsics"] = camera_infos["intrinsics"][test_cmr_ids]
        camera_infos["extrinsics"] = camera_infos["extrinsics"][test_cmr_ids]

    cameras = {}
    for i in range(len(camera_infos["ids"])):
        cam_id = camera_infos["ids"][i]
        intrinsic = camera_infos["intrinsics"][i]

        if wo_cxcy:
            intrinsic[0, 2] = camera_infos["shape"][i, 1] / 2
            intrinsic[1, 2] = camera_infos["shape"][i, 0] / 2

        extrinsic = camera_infos["extrinsics"][i]
        camera = OpenCVCamera(K=intrinsic, Rt=extrinsic, cols=cols, rows=rows, viewer=viewer)
        cameras[str(cam_id)] = camera

    cameras = dict(sorted(cameras.items()))

    material = Material(ambient=0.5, diffuse=0.0, specular=0.0)
    if from_humanrf:
        mesh_path_list = [(mesh_folder / f'mesh_{i:06d}_smooth_{mesh_res}.obj')
                          for i in range(frame_0, frame_end, interval)]
        meshes = VariableTopologyMeshes.from_plys(mesh_path_list, vertex_scale=1, material=material)
        label = "_humanrf"

        for f_idx in tqdm(range(frame_0, frame_end, interval)):
            frame_out_dir = gstar_folder / f"{f_idx:04d}/"
            mask_dir = frame_out_dir / f"masks{label}/"
            os.makedirs(mask_dir, exist_ok=True)
            depth_dir = frame_out_dir / f"depth{label}/"
            os.makedirs(depth_dir, exist_ok=True)
    else:
        meshes = VariableTopologyMeshes.from_directory(mesh_folder, vertex_scale=0.001, material=material)
        label = ""

    viewer.scene.add(meshes)
    n_frames = meshes.n_frames

    for cmr_idx in range(len(cameras)):
        cam_name = camera_infos["ids"][cmr_idx]
        camera = cameras[str(cam_name)]
        print(f'\nStart to process camera {cam_name} as new camera {cmr_idx}')
        viewer.set_temp_camera(camera)

        for i_mesh in tqdm(range(0, n_frames, interval)):
            meshes.current_frame_id = i_mesh
            i = i_mesh + frame_0

            mask = np.array(viewer.get_mask())
            mask = mask[:shape[cmr_idx, 0], :shape[cmr_idx, 1]]
            mask = np.uint8((mask.mean(axis=2) > 50) * 255)
            cv2.imwrite(osp.join(gstar_folder, f"{i:04d}/masks{label}/img_{int(cmr_idx):04d}_alpha.png"), mask)

            depth = np.array(viewer.get_depth())
            depth = depth[:shape[cmr_idx, 0], :shape[cmr_idx, 1]]
            depth_vis = np.minimum(10, depth)
            depth_vis = np.uint8(depth_vis / 10 * 255.0)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            np.savez_compressed(osp.join(gstar_folder, f"{i:04d}/depth{label}/img_{int(cmr_idx):04d}_depth.npz"), depth=depth)
            cv2.imwrite(osp.join(gstar_folder, f"{i:04d}/depth{label}/img_{int(cmr_idx):04d}_depth.jpg"), depth_vis)


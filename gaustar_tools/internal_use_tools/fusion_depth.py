
import os
import open3d as o3d
import numpy as np
import cv2
from tqdm import tqdm

from gaustar_scene.gs_model import GaussianSplattingWrapper
from gaustar_trainers.refined_mesh import get_depth_edge, to_cam_open3d


def extract_mesh_fusion(out_dir, nerfmodel: GaussianSplattingWrapper,
                        voxel_size=0.006, sdf_trunc=0.02, depth_trunc=6, simplify_face_num=0,
                        mask_backgrond=True, save_dir=None, smooth=False, remove_depth_edge=True):
    """
    Perform TSDF fusion given a fixed depth range, used in the paper.

    voxel_size: the voxel size of the volume
    sdf_trunc: truncation value
    depth_trunc: maximum depth range, should depended on the scene's scales
    mask_backgrond: whether to mask backgroud, only works when the dataset have masks

    return o3d.mesh
    """
    print("Running tsdf volume integration ...")
    print(f'voxel_size: {voxel_size}')
    print(f'sdf_trunc: {sdf_trunc}')
    print(f'depth_truc: {depth_trunc}')

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    os.makedirs(out_dir, exist_ok=True)
    # save_dir = out_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cmr_list = to_cam_open3d(nerfmodel.cam_list)
    cmr_num = len(cmr_list)

    for cmr_i in tqdm(range(cmr_num)):
        cam_o3d = cmr_list[cmr_i]
        rgb = nerfmodel.cam_list[cmr_i].original_image.permute(1, 2, 0).cpu().numpy()
        depth = nerfmodel.cam_list[cmr_i].original_depth.permute(1, 2, 0).cpu().numpy()

        # if we have mask provided, use it
        if mask_backgrond:
            bg_mask = depth > 7
            # bg_mask = alpha < 0.5
            depth[bg_mask] = 0

        if remove_depth_edge:
            edge_depth = get_depth_edge(depth, ker_size=3)
            edge_vis = np.minimum(edge_depth / edge_depth.max() * 1000, 1)
            if save_dir:
                cv2.imwrite(save_dir + f"edge_vis_{cmr_i:04d}.jpg", edge_vis * 255.0)
            edge_mask = edge_vis > 0.5
            depth[edge_mask] = 0

        if save_dir:
            cv2.imwrite(save_dir + f"color_{cmr_i:06d}.jpg", rgb[..., ::-1] * 255)
            render_depth_vis = np.uint8(depth / 10 * 255.0)
            render_depth_vis = cv2.applyColorMap(render_depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(save_dir + f"depth_{cmr_i:06d}.jpg", render_depth_vis)

        # make open3d rgbd
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.asarray(rgb * 255, order="C", dtype=np.uint8)),
            o3d.geometry.Image(np.asarray(depth, order="C")),
            depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )

        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

    mesh = volume.extract_triangle_mesh()
    if smooth:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=15)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
    if simplify_face_num > 0:
        mesh = mesh.simplify_quadric_decimation(simplify_face_num)
    return mesh


if __name__ == "__main__":

    gs_checkpoint_path = "../gs_output/track_240906_Take8/0010/"
    data_dir = "/media/dalco/Data_Chengwei/SuGaR/data/track_240906_Take8/"

    for f_idx in range(89, 90, 1):
        source_path = data_dir + f"{f_idx:04d}/"
        nerfmodel = GaussianSplattingWrapper(
            source_path=source_path,
            output_path=gs_checkpoint_path,
            iteration_to_load=1,
            load_gt_images=True,
            eval_split=False,
            eval_split_interval=0,
            from_humanrf=True,
        )

        out_dir = "/media/dalco/Data_Chengwei/humanrf/out/mocap_240906_Take8/results/fusion/"
        fusion_mesh = extract_mesh_fusion(out_dir, nerfmodel,
                                          # save_dir=(mesh_output_dir + "extract/"),
                                          # simplify_face_num=np.array(o3d_mesh.triangles).shape[0],
                                          simplify_face_num=200_000,
                                          smooth=False)
        o3d.io.write_triangle_mesh(out_dir + f"extract_{f_idx:04d}_smooth_re6.obj", fusion_mesh)


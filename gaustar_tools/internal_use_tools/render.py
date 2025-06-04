import cv2
import torch.cuda
import os
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
import trimesh

from gaustar_scene.gs_model import GaussianSplattingWrapper
from gaustar_scene.sugar_model import SuGaR
from gaustar_utils.spherical_harmonics import SH2RGB

from rich.console import Console
from scipy.spatial.transform import Rotation

max_depth = 10


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def extract_mesh_and_texture_from_refined_sugar(args):
    CONSOLE = Console(width=120)

    n_skip_images_for_eval_split = 8

    # --- Scene data parameters ---
    source_path = args.scene_path
    use_train_test_split = args.eval

    # --- Vanilla 3DGS parameters ---
    iteration_to_load = args.iteration_to_load
    gs_checkpoint_path = args.checkpoint_path

    # --- Fine model parameters ---
    refined_model_path = args.refined_model_path
    n_gaussians_per_surface_triangle = args.n_gaussians_per_surface_triangle

    sugar_mesh_path = args.mesh_path

    CONSOLE.print('==================================================')
    CONSOLE.print("Starting extracting texture from refined SuGaR model:")
    CONSOLE.print('Scene path:', source_path)
    CONSOLE.print('Iteration to load:', iteration_to_load)
    CONSOLE.print('Vanilla 3DGS checkpoint path:', gs_checkpoint_path)
    CONSOLE.print('Refined model path:', refined_model_path)
    CONSOLE.print('Coarse mesh path:', sugar_mesh_path)
    # CONSOLE.print('Mesh output directory:', mesh_output_dir)
    # CONSOLE.print('Mesh save path:', mesh_save_path)
    CONSOLE.print('Number of gaussians per surface triangle:', n_gaussians_per_surface_triangle)
    CONSOLE.print('==================================================')

    # Set the GPU
    torch.cuda.set_device(args.gpu)

    # ==========================

    # --- Loading Vanilla 3DGS model ---
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("Gaussian splatting checkpoint path:", gs_checkpoint_path)
    CONSOLE.print(f"\nLoading Vanilla 3DGS model config {gs_checkpoint_path}...")

    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=args.load_gt,  # TODO: Check
        eval_split=use_train_test_split,
        eval_split_interval=n_skip_images_for_eval_split,
        )
    CONSOLE.print("Vanilla 3DGS Loaded.")
    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')
    CONSOLE.print(len(nerfmodel.gaussians._xyz) / 1e6, "M gaussians detected.")

    # ZCW change
    nerfmodel.gaussians.active_sh_degree = 2

    # --- Loading coarse mesh ---
    o3d_mesh = o3d.io.read_triangle_mesh(sugar_mesh_path)

    # --- Loading refined SuGaR model ---
    checkpoint = torch.load(refined_model_path, map_location=nerfmodel.device)
    if '_delta_t' or '_delta_r' in checkpoint['state_dict']:
        use_delta = True
    else:
        use_delta = False
    refined_sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint['state_dict']['_points'],
        colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
        initialize=False,
        sh_levels=nerfmodel.gaussians.active_sh_degree+1,
        keep_track_of_knn=False,
        knn_to_track=0,
        beta_mode='average',
        surface_mesh_to_bind=o3d_mesh,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
        delta_allowed=use_delta,
        )
    # refined_sugar.unbind_surface_mesh()
    if use_delta:
        refined_sugar.loose_bind()
    refined_sugar.load_state_dict(checkpoint['state_dict'])
    refined_sugar.eval()

    mesh = trimesh.load_mesh(sugar_mesh_path)
    mesh_center = np.average(mesh.bounds, axis=0)
    # mesh_center[0] = 0.25
    mesh_center_trans = np.identity(4)
    mesh_center_trans_inv = np.identity(4)
    mesh_center_trans[:3, 3] = -mesh_center
    mesh_center_trans_inv[:3, 3] = mesh_center

    camera_infos = dict(np.load(args.cmr_path))
    extrinsic = camera_infos["extrinsics"][args.base_cmr_id]
    extr4x4_mat = np.identity(4)
    extr4x4_mat[:3, :4] = extrinsic

    # ZCW rendering
    if args.render_results and 1:
        render_dir = output_dir + f"render_rotating/"
        os.makedirs(render_dir, exist_ok=True)
        with (torch.no_grad()):
            delta = 3
            for theta in tqdm(range(0, 360, delta)):
                # RGB
                r_mat = np.identity(4)
                r_mat[:3, :3] = Rotation.from_euler("zyx", [0, theta, 0], degrees=True).as_matrix()
                # rot_extr = np.matmul(extr4x4_mat, r_mat)
                rot_extr = extr4x4_mat @ mesh_center_trans_inv @ r_mat @ mesh_center_trans

                rgb = refined_sugar.render_image_gaussian_rasterizer(
                    camera_indices=args.base_cmr_id,
                    overwrite_extr=rot_extr,
                    bg_color=[1.0, 1.0, 1.0],
                    sh_deg=nerfmodel.gaussians.active_sh_degree,
                    compute_color_in_rasterizer=True,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=False,
                    use_same_scale_in_all_directions=False,
                ).clamp(min=0., max=1.).contiguous()

                render_img = rgb.cpu().numpy()[..., ::-1]
                cv2.imwrite(render_dir + f"/render_{args.f_idx:04d}_{(theta//delta)}.jpg", render_img * 255.0)


scene_path = "/mnt/euler/SUGAR/data/mocap/track_241028_Take4/"
gs_ckpt_path = "/mnt/euler/SUGAR/SuGaR/gs_output/track_241028_Take4/0000/"
f_idx = 220
output_dir = "/mnt/server02/GSTAR/track_241028T4/"
refinement_iterations = 2000
base_cmr_id = 30

refined_mesh_args = AttrDict({
    'scene_path': scene_path + f"{f_idx:04d}/",
    'iteration_to_load': 1,  # ZCW debug refinement_iterations
    'checkpoint_path': gs_ckpt_path,
    'mesh_path': output_dir + f"{f_idx:04d}/color_mesh.obj",
    'refined_model_path': output_dir + f"{f_idx:04d}/{refinement_iterations}.pt",
    'cmr_path': scene_path + "rgb_cameras.npz",
    'output_dir': output_dir,  # "refined_mesh/",
    'n_gaussians_per_surface_triangle': 6,
    'eval': False,
    'gpu': 0,
    'UV_texture': False,
    'render_results': True,
    'load_gt': False,
    'save_diff': False,
    'base_cmr_id':base_cmr_id,
    'f_idx':f_idx,
})

extract_mesh_and_texture_from_refined_sugar(refined_mesh_args)

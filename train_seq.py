import argparse
import torch.cuda
import os
import shutil

from gaustar_utils.general_utils import str2bool
from gaustar_trainers.refine import refined_training
from gaustar_trainers.refined_mesh import forward_rendering_and_mesh_update
from gaustar_tools.warp_mesh import warp_mesh_using_flow

class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full SuGaR model.')

    # Data and vanilla 3DGS checkpoint
    parser.add_argument('-s', '--scene_path',
                        type=str,
                        help='(Required) path to the scene data to use.')
    parser.add_argument('-c', '--checkpoint_path',
                        type=str,
                        default=None,
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-o', '--output_path',
                        type=str,
                        help='(Required) path to the save directory.')
    parser.add_argument('-i', '--iteration_to_load',
                        type=int, default=1,
                        help='iteration to load.')

    # Extract mesh
    parser.add_argument('-b', '--bboxmin', type=str, default=None,
                        help='Min coordinates to use for foreground.')
    parser.add_argument('-B', '--bboxmax', type=str, default=None,
                        help='Max coordinates to use for foreground.')

    # Parameters for refined SuGaR
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=6,
                        help='Number of gaussians per triangle.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=2_000,
                        help='Number of refinement iterations.')

    # (Optional) Parameters for textured mesh extraction
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False,
                        help='If True, postprocess the mesh by removing border triangles with low-density. '
                        'This step takes a few minutes and is not needed in general, as it can also be risky. '
                        'However, it increases the quality of the mesh in some cases, especially when an object is visible only from one side.')
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1,
                        help='Threshold to use for postprocessing the mesh.')
    parser.add_argument('--postprocess_iterations', type=int, default=5,
                        help='Number of iterations to use for postprocessing the mesh.')

    # (Optional) PLY file export
    parser.add_argument('--export_ply', type=str2bool, default=True,
                        help='If True, export a ply file with the refined 3D Gaussians at the end of the training. '
                        'This file can be large (+/- 500MB), but is needed for using the dedicated viewer. Default is True.')

    # Evaluation split
    parser.add_argument('--eval', type=str2bool, default=False, help='Use eval split.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')

    parser.add_argument('--frame_0', type=int, default=0)
    parser.add_argument('--frame_end', type=int, default=0)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--base_mesh', type=str, default="MS_100k.obj")

    parser.add_argument("--disable_mesh_update", action="store_true")
    parser.add_argument("--from_humanrf", action="store_true")
    parser.add_argument("--SH_reg", action="store_true")

    # Parse arguments
    args = parser.parse_args()

    frame_0 = args.frame_0
    frame_end = args.frame_end
    interval = args.interval
    enable_mesh_update = not args.disable_mesh_update
    from_humanrf = args.from_humanrf
    SH_reg = args.SH_reg
    force_watertight = False

    args.scene_path = os.path.join(args.scene_path, '')
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(args.scene_path, 'gs_out/')
    args.output_path = os.path.join(args.output_path, '')

    if enable_mesh_update:
        loose_bind_from = args.refinement_iterations // 2
    else:
        loose_bind_from = 999999

    tracking_data = None
    # tracking_data = args.output_path + f"{50:04d}/face_corr.npz"
    for f_idx in range(frame_0, frame_end, interval):

        if f_idx == frame_0:
            coarse_mesh_path = args.scene_path + args.base_mesh
            coarse_mesh_from_MS = True
            pre_checkpoint_path = None
            ref_mesh_path = args.scene_path + args.base_mesh
            ref_edge_loss_factor = 1000
            ref_area_loss_factor = 5000
            NC_factor = 0.5
        else:
            coarse_mesh_path = args.output_path + f"{f_idx:04d}/coarse_mesh/warp_smooth.obj"
            coarse_mesh_from_MS = False
            pre_checkpoint_path = args.output_path + f"{(f_idx-interval):04d}/{args.refinement_iterations}.pt"
            ref_mesh_path = args.output_path + f"{frame_0:04d}/color_mesh.obj"
            ref_edge_loss_factor = 1000
            ref_area_loss_factor = 1000
            NC_factor = 0.5

        label = ""

        # ----- Refine SuGaR -----
        refined_args = AttrDict({
            'scene_path': args.scene_path + f"{f_idx:04d}/",
            'checkpoint_path': args.checkpoint_path,
            'mesh_path': coarse_mesh_path,
            'cmr_path': args.scene_path + "rgb_cameras.npz",
            'output_dir': args.output_path + f"{f_idx:04d}{label}/",
            'iteration_to_load': args.iteration_to_load,
            'normal_consistency_factor': NC_factor,
            'gaussians_per_triangle': args.gaussians_per_triangle,
            'refinement_iterations': args.refinement_iterations,
            'bboxmin': args.bboxmin,
            'bboxmax': args.bboxmax,
            'export_ply': args.export_ply,
            'eval': args.eval,
            'gpu': args.gpu,
            'pre_checkpoint_path': pre_checkpoint_path,
            'ref_mesh_path': ref_mesh_path,
            'tracking_data': tracking_data,
            'ref_edge_loss_factor': ref_edge_loss_factor,
            'ref_area_loss_factor': ref_area_loss_factor,
            'coarse_mesh_from_MS': coarse_mesh_from_MS,
            'densifier_detection_only': False,
            'loose_bind_from': loose_bind_from,
            'from_humanrf': from_humanrf,
            'SH_reg': SH_reg,
        })
        refined_sugar_path, enable_unbind = refined_training(refined_args)
        # refined_sugar_path = args.output_path + f"{f_idx:04d}/{args.refinement_iterations}.pt"
        # enable_unbind = True


        # ----- Extract mesh and texture from refined SuGaR -----
        torch.cuda.empty_cache()
        refined_mesh_args = AttrDict({
            'scene_path': args.scene_path + f"{f_idx:04d}/",
            'iteration_to_load': args.iteration_to_load,  # ZCW debug refinement_iterations
            'checkpoint_path': args.checkpoint_path,
            'mesh_path': coarse_mesh_path,
            'refined_model_path': refined_sugar_path,
            'cmr_path': args.scene_path + "rgb_cameras.npz",
            'mesh_output_dir': args.output_path + f"{f_idx:04d}{label}/",
            'n_gaussians_per_surface_triangle': args.gaussians_per_triangle,
            'eval': args.eval,
            'gpu': args.gpu,
            'postprocess_mesh': args.postprocess_mesh,
            'postprocess_density_threshold': args.postprocess_density_threshold,
            'postprocess_iterations': args.postprocess_iterations,
            'UV_texture': False,
            'mesh_extraction': True,
            'render_results': 'wd',
            'load_gt': False,
            'save_diff': False,
            'enable_mesh_update': enable_mesh_update,
            'enable_unbind': enable_unbind,
            'force_watertight': force_watertight,
            'from_humanrf': from_humanrf,
        })
        refined_mesh_path = forward_rendering_and_mesh_update(refined_mesh_args)
        # refined_mesh_path = args.output_path + f"{f_idx:04d}/color_mesh.obj"

        # if enable_mesh_update:
        if enable_mesh_update and os.path.exists(args.output_path + f"{f_idx:04d}/updated_mesh.obj"):
            tracking_data = args.output_path + f"{f_idx:04d}/face_corr.npz"

            refined_args = AttrDict({
                'scene_path': args.scene_path + f"{f_idx:04d}/",
                'checkpoint_path': args.checkpoint_path,
                'mesh_path': args.output_path + f"{f_idx:04d}/updated_mesh.obj",
                'cmr_path': args.scene_path + "rgb_cameras.npz",
                'output_dir': args.output_path + f"{f_idx:04d}{label}_update/",
                'iteration_to_load': args.iteration_to_load,
                'normal_consistency_factor': NC_factor,
                'gaussians_per_triangle': args.gaussians_per_triangle,
                'refinement_iterations': args.refinement_iterations // 2,
                'bboxmin': args.bboxmin,
                'bboxmax': args.bboxmax,
                'export_ply': args.export_ply,
                'eval': args.eval,
                'gpu': args.gpu,
                'pre_checkpoint_path': None,
                'ref_mesh_path': ref_mesh_path,
                'tracking_data': tracking_data,
                'ref_edge_loss_factor': ref_edge_loss_factor,
                'ref_area_loss_factor': ref_area_loss_factor,
                'coarse_mesh_from_MS': coarse_mesh_from_MS,
                'densifier_detection_only': False,
                'loose_bind_from': 999999,
                'from_humanrf': from_humanrf,
                'SH_reg': SH_reg,
            })
            refined_sugar_path, _ = refined_training(refined_args)
            # refined_sugar_path = args.output_path + f"{f_idx:04d}_update/{(args.refinement_iterations // 2)}.pt"

            torch.cuda.empty_cache()
            refined_mesh_args = AttrDict({
                'scene_path': args.scene_path + f"{f_idx:04d}/",
                'iteration_to_load': args.iteration_to_load,
                'checkpoint_path': args.checkpoint_path,
                'mesh_path': args.output_path + f"{f_idx:04d}/updated_mesh.obj",
                'refined_model_path': refined_sugar_path,
                'cmr_path': args.scene_path + "rgb_cameras.npz",
                'mesh_output_dir': args.output_path + f"{f_idx:04d}{label}_update/",
                'n_gaussians_per_surface_triangle': args.gaussians_per_triangle,
                'eval': args.eval,
                'gpu': args.gpu,
                'postprocess_mesh': args.postprocess_mesh,
                'postprocess_density_threshold': args.postprocess_density_threshold,
                'postprocess_iterations': args.postprocess_iterations,
                'UV_texture': False,
                'mesh_extraction': True,
                'render_results': 'w',
                'load_gt': False,
                'save_diff': False,
                'enable_mesh_update': False,
                'is_loose_bind': False,
                'from_humanrf': from_humanrf,
            })
            refined_mesh_path = forward_rendering_and_mesh_update(refined_mesh_args)

        if f_idx < frame_end - interval:
            # ----- Warp mesh using optical flow for next frame initialization -----
            warp_mesh_using_flow(refined_mesh_path, args.scene_path, args.output_path,
                                 f_idx, interval=interval, save_inter=False, from_humanrf=from_humanrf)
            # os.makedirs(args.output_path + f"{(f_idx+interval):04d}/coarse_mesh/", exist_ok=True)
            # shutil.copy(refined_mesh_path, args.output_path + f"{(f_idx+interval):04d}/coarse_mesh/warp_smooth.obj")
        else:
            print("---GsuSTAR Finish---")

import argparse
import torch.cuda
import os

from gaustar_utils.general_utils import str2bool
from gaustar_trainers.refined_mesh import forward_rendering_and_mesh_update


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

    parser.add_argument("--from_humanrf", action="store_true")

    # Parse arguments
    args = parser.parse_args()

    frame_0 = args.frame_0
    frame_end = args.frame_end
    interval = args.interval
    enable_mesh_update = False
    from_humanrf = args.from_humanrf
    force_watertight = False

    args.scene_path = os.path.join(args.scene_path, '')
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(args.scene_path, 'gs_out/')
    args.output_path = os.path.join(args.output_path, '')

    for f_idx in range(frame_0, frame_end, interval):

        coarse_mesh_path = args.output_path + f"{f_idx:04d}/color_mesh.obj"
        label = ""
        refined_sugar_path = args.output_path + f"{f_idx:04d}/{args.refinement_iterations}.pt"

        # ----- Extract mesh and texture from refined SuGaR -----
        torch.cuda.empty_cache()
        refined_mesh_args = AttrDict({
            'scene_path': args.scene_path + f"{f_idx:04d}/",
            'iteration_to_load': args.iteration_to_load,  # ZCW debug refinement_iterations
            'checkpoint_path': args.checkpoint_path,
            'mesh_path': coarse_mesh_path,
            'refined_model_path': refined_sugar_path,
            'cmr_path': args.scene_path + "rgb_cameras.npz",
            # 'mesh_output_dir': args.output_path + f"{label}/",  # "refined_mesh/",
            'mesh_output_dir': args.output_path + f"{f_idx:04d}{label}/",  # "refined_mesh/",
            'n_gaussians_per_surface_triangle': args.gaussians_per_triangle,
            'eval': args.eval,
            'gpu': args.gpu,
            'postprocess_mesh': args.postprocess_mesh,
            'postprocess_density_threshold': args.postprocess_density_threshold,
            'postprocess_iterations': args.postprocess_iterations,
            'UV_texture': False,
            'mesh_extraction': True,
            'render_results': 'bd',  # 'wd' 'gd' remove 'd' if do not need depth
            'load_gt': False,
            'save_diff': False,
            'enable_mesh_update': enable_mesh_update,
            'enable_unbind': False,
            'force_watertight': force_watertight,
            # 'label': label,
            'from_humanrf': from_humanrf,
        })
        refined_mesh_path = forward_rendering_and_mesh_update(refined_mesh_args)
        # refined_mesh_path = args.output_path + f"{f_idx:04d}/color_mesh.obj"

import os
import cv2
import numpy as np
import torch
import open3d as o3d
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency

from gaustar_scene.gs_model import GaussianSplattingWrapper
from gaustar_scene.sugar_model import SuGaR, convert_refined_sugar_into_gaussians
from gaustar_scene.sugar_optimizer import OptimizationParams, SuGaROptimizer
from gaustar_scene.sugar_densifier import SuGaRDensifier
from gaustar_utils.loss_utils import ssim, l1_loss, l2_loss

from rich.console import Console
import time

import json
import trimesh

from pytorch3d.structures import Meshes
from gaustar_trainers.refined_mesh import detect_topo_err


class opti_config:
    mask_loss_factor = 1
    mask_loss_from = 0
    depth_loss_factor = 0.1
    depth_loss_from = 0
    sh_reg_loss_factor = 1
    use_opacity_reg = True
    min_opacity = 0.8
    loose_bind_factor_t = 100
    loose_bind_factor_r = 1
    use_margin = True  # For ActorHQ dataset, as cx cy != shape / 2
    print_training_info = True


def refined_training(args):
    CONSOLE = Console(width=120)
    cfg = opti_config()

    # ====================Parameters====================

    num_device = args.gpu
    detect_anomaly = False
    downscale_resolution_factor = 1  # 2, 4

    # -----Model parameters-----
    n_skip_images_for_eval_split = 8

    freeze_gaussians = False
    # initialize_from_trained_3dgs = False  # True or False
    no_rendering = freeze_gaussians

    n_points_at_start = None  # If None, takes all points in the SfM point cloud
    learnable_positions = True  # True in 3DGS
    use_same_scale_in_all_directions = False  # Should be False
    sh_levels = 3  # SH: 3 or 4

    # -----Radiance Mesh-----
    triangle_scale=1.
        
    # -----Rendering parameters-----
    compute_color_in_rasterizer = False
        
    # -----Optimization parameters-----

    # Learning rates and scheduling
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001

    # Densifier (for positional gradient)
    use_densifier = True
    if use_densifier:
        heavy_densification = False

        densify_from_iter = 99  # 500  # Maybe reduce this, since we have a better initialization?
        densify_until_iter = 7000  # 7000

        if heavy_densification:
            densification_interval = 50  # 100
            opacity_reset_interval = 3000  # 3000

            densify_grad_threshold = 0.0001  # 0.0002
            densify_screen_size_threshold = 20
            prune_opacity_threshold = 0.005
            densification_percent_distinction = 0.01
        else:
            densification_interval = 200  # 100
            opacity_reset_interval = 3000  # 3000

            densify_grad_threshold = 0.0001 * 0.4  # 0.0002
            densify_screen_size_threshold = 20
            prune_opacity_threshold = 0.005
            densification_percent_distinction = 0.01

    train_num_images_per_batch = 1  # 1 for full images

    # Loss functions
    loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
    if loss_function == 'l1+dssim':
        dssim_factor = 0.2
        
    use_surface_losses = True
    learn_surface_mesh_positions = True
    learn_surface_mesh_opacity = True
    learn_surface_mesh_scales = True
    # n_gaussians_per_surface_triangle=6  # 1, 3, 4 or 6
    use_surface_mesh_laplacian_smoothing_loss = False
    surface_mesh_laplacian_smoothing_factor = None
    if use_surface_mesh_laplacian_smoothing_loss:
        surface_mesh_laplacian_smoothing_method = "uniform"  # "cotcurv", "cot", "uniform"
        surface_mesh_laplacian_smoothing_factor = 5.  # 0.1

    use_surface_mesh_normal_consistency_loss = True
    use_densifier = False
    regularize = False

    loose_bind_from = 1000
    if hasattr(args, "loose_bind_from"):
        loose_bind_from = args.loose_bind_from
        mesh_prop_for_detect = 20

    densifier_detection_only = args.densifier_detection_only
    if densifier_detection_only:
        use_densifier = True
        densify_until_iter = loose_bind_from + 1

    if args.SH_reg:
        sh_reg_loss_from = 0
    else:
        sh_reg_loss_from = 999999

    area_reg_loss_factor = 0.1
    area_reg_loss_from = 999999

    pre_color_fix = False

    max_gaussian_scale = 0.003
    min_gaussian_scale = 0.0003

    do_sh_warmup = True  # Should be True
    if do_sh_warmup:
        sh_warmup_every = args.refinement_iterations // 4
        current_sh_levels = 1
    else:
        current_sh_levels = sh_levels

    # -----Log and save-----
    print_loss_every_n_iterations = 50
    save_model_every_n_iterations = 1_000_000 # 500, 1_000_000  # TODO
    save_milestones = [args.refinement_iterations]  #  [7_000, 15_000]  # 2000,

    # ====================End of parameters====================

    if args.output_dir is None:
        if len(args.scene_path.split("/")[-1]) > 0:
            args.output_dir = os.path.join("./output/refined", args.scene_path.split("/")[-1])
        else:
            args.output_dir = os.path.join("./output/refined", args.scene_path.split("/")[-2])
            
    # Bounding box
    if args.bboxmin is None:
        use_custom_bbox = False
    else:
        if args.bboxmax is None:
            raise ValueError("You need to specify both bboxmin and bboxmax.")
        use_custom_bbox = True
        
        # Parse bboxmin
        if args.bboxmin[0] == '(':
            args.bboxmin = args.bboxmin[1:]
        if args.bboxmin[-1] == ')':
            args.bboxmin = args.bboxmin[:-1]
        args.bboxmin = tuple([float(x) for x in args.bboxmin.split(",")])
        
        # Parse bboxmax
        if args.bboxmax[0] == '(':
            args.bboxmax = args.bboxmax[1:]
        if args.bboxmax[-1] == ')':
            args.bboxmax = args.bboxmax[:-1]
        args.bboxmax = tuple([float(x) for x in args.bboxmax.split(",")])
            
    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    surface_mesh_to_bind_path = args.mesh_path
    iteration_to_load = args.iteration_to_load
    
    surface_mesh_normal_consistency_factor = args.normal_consistency_factor
    n_gaussians_per_surface_triangle = args.gaussians_per_triangle
    num_iterations = args.refinement_iterations

    sugar_checkpoint_path = args.output_dir
        
    if use_custom_bbox:
        fg_bbox_min = args.bboxmin
        fg_bbox_max = args.bboxmax
    
    use_eval_split = args.eval
    
    export_ply_at_the_end = args.export_ply
    
    CONSOLE.print("-----Parsed parameters-----")
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("   > Content:", len(os.listdir(source_path)))
    CONSOLE.print("Gaussian Splatting checkpoint path:", gs_checkpoint_path)
    CONSOLE.print("   > Content:", len(os.listdir(gs_checkpoint_path)))
    CONSOLE.print("Checkpoint path:", sugar_checkpoint_path)
    CONSOLE.print("Surface mesh to bind to:", surface_mesh_to_bind_path)
    CONSOLE.print("Normal consistency factor:", surface_mesh_normal_consistency_factor)
    CONSOLE.print("Number of gaussians per surface triangle:", n_gaussians_per_surface_triangle)
    if use_custom_bbox:
        CONSOLE.print("Foreground bounding box min:", fg_bbox_min)
        CONSOLE.print("Foreground bounding box max:", fg_bbox_max)
    CONSOLE.print("Export ply at the end:", export_ply_at_the_end)
    CONSOLE.print("----------------------------")
    
    # Setup device
    torch.cuda.set_device(num_device)
    CONSOLE.print("Using device:", num_device)
    device = torch.device(f'cuda:{num_device}')
    CONSOLE.print(torch.cuda.memory_summary())
    
    torch.autograd.set_detect_anomaly(detect_anomaly)
    
    # Creates save directory if it does not exist
    os.makedirs(sugar_checkpoint_path, exist_ok=True)
    
    # ====================Load NeRF model and training data====================

    # Load Gaussian Splatting checkpoint 
    CONSOLE.print(f"\nLoading config {gs_checkpoint_path}...")
    if use_eval_split:
        CONSOLE.print("Performing train/eval split...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=True,
        eval_split=use_eval_split,
        eval_split_interval=n_skip_images_for_eval_split,
        from_humanrf=args.from_humanrf,
        )

    cmr_npz = dict(np.load(args.cmr_path))
    if cfg.use_margin:
        cmr_num = cmr_npz["intrinsics"].shape[0]
        cmr_margin_list = np.ones((cmr_num, 4), dtype=np.int32)  # left right top bottom
        for cmr_i in range(cmr_num):
            cx = cmr_npz["intrinsics"][cmr_i, 0, 2]
            cy = cmr_npz["intrinsics"][cmr_i, 1, 2]
            rows = cmr_npz["shape"][cmr_i, 0]
            cols = cmr_npz["shape"][cmr_i, 1]
            if cx < cols / 2:
                cmr_margin_list[cmr_i, 0] = int(cols / 2 - cx) + 1 # left
            else:
                cmr_margin_list[cmr_i, 1] = int(cx - cols / 2) + 1 # right
            if cy < rows / 2:
                cmr_margin_list[cmr_i, 2] = int(rows / 2 - cy) + 1 # top
            else:
                cmr_margin_list[cmr_i, 3] = int(cy - rows / 2) + 1 # bottom

    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')

    if downscale_resolution_factor != 1:
       nerfmodel.downscale_output_resolution(downscale_resolution_factor)
    CONSOLE.print(f'\nCamera resolution scaled to '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_height} x '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_width}'
          )

    # surface_mesh_to_bind_full_path = os.path.join('./results/meshes/', surface_mesh_to_bind_path)
    surface_mesh_to_bind_full_path = surface_mesh_to_bind_path
    CONSOLE.print(f'\nLoading mesh to bind to: {surface_mesh_to_bind_full_path}...')
    o3d_mesh = o3d.io.read_triangle_mesh(surface_mesh_to_bind_full_path)
    CONSOLE.print("Mesh to bind to loaded.")

    if args.ref_mesh_path:
        CONSOLE.print(f'\nLoading reference mesh: {args.ref_mesh_path}...')
        ref_mesh = o3d.io.read_triangle_mesh(args.ref_mesh_path)
        ref_mesh = Meshes(
            verts=[torch.tensor(np.asarray(ref_mesh.vertices)).float().to(nerfmodel.device)],
            faces=[torch.tensor(np.asarray(ref_mesh.triangles)).to(nerfmodel.device)]
        )

        edge_iso_loss_factor = args.ref_edge_loss_factor
        edge_iso_loss_from = 0
        area_iso_loss_factor = args.ref_area_loss_factor
        area_iso_loss_from = 0

        ref_verts_edges = ref_mesh.verts_packed()[ref_mesh.edges_packed()]
        ref_v0, ref_v1 = ref_verts_edges.unbind(1)
        ref_edge_len = (ref_v0 - ref_v1).norm(dim=1, p=2)
        ref_area = ref_mesh.faces_areas_packed()
        max_gaussian_scalar = 5  # pre: 0.8 or 0.7
        min_gaussian_scalar = 0.1  # 0.08
        if args.coarse_mesh_from_MS:
            ref_edge_len = ref_edge_len * 0.98
            ref_area = ref_area * 0.95
        max_gaussian_scale = ref_edge_len.mean().item() * max_gaussian_scalar
        min_gaussian_scale = ref_edge_len.mean().item() * min_gaussian_scalar

    if args.tracking_data:
        CONSOLE.print(f'\nLoading tracking data: {args.tracking_data}...')
        tracking_data = np.load(args.tracking_data)
        edge_iso_loss_factor = args.ref_edge_loss_factor
        edge_iso_loss_from = 999999
        area_iso_loss_factor = args.ref_area_loss_factor
        area_iso_loss_from = 0
        ref_area = torch.from_numpy(np.float32(tracking_data['ref_area'])).to(nerfmodel.device)

        track_face_mask = tracking_data['track_face_mask']
        track_face_num = track_face_mask.sum()

    pre_checkpoint = None
    if args.pre_checkpoint_path:
        pre_checkpoint = torch.load(args.pre_checkpoint_path, map_location=nerfmodel.device)
        if os.path.exists(os.path.dirname(args.pre_checkpoint_path) + f"/updated_mesh.obj"):
            pre_ckpt_update = True
        else:
            pre_ckpt_update = False

    pre_ckpt_color_init = False
    pre_ckpt_scale_R_init = False
    pre_ckpt_opacity_init = False
    pre_ckpt_init = pre_ckpt_color_init or pre_ckpt_scale_R_init or pre_ckpt_opacity_init
    
    # ====================Initialize SuGaR model====================
    # Construct SuGaR model
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=None,
        colors=None,
        initialize=True,
        sh_levels=sh_levels,
        learnable_positions=learnable_positions,
        triangle_scale=triangle_scale,
        keep_track_of_knn=regularize,
        knn_to_track=0,
        beta_mode=None,
        freeze_gaussians=freeze_gaussians,
        surface_mesh_to_bind=o3d_mesh,
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=learn_surface_mesh_positions,
        learn_surface_mesh_opacity=learn_surface_mesh_opacity,
        learn_surface_mesh_scales=learn_surface_mesh_scales,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
        max_gaussian_scale=max_gaussian_scale,
        min_gaussian_scale=min_gaussian_scale,
    )

    if pre_checkpoint and pre_ckpt_init:
        sugar_dict = sugar.state_dict()
        update_name = []
        if pre_ckpt_color_init:
            update_name += ['_sh_coordinates_dc', '_sh_coordinates_rest']
        if pre_ckpt_scale_R_init:
            update_name += ['_scales', '_quaternions']
        if pre_ckpt_opacity_init:
            update_name.append('all_densities')
        if '_delta_t' in pre_checkpoint['state_dict']:
            update_name.append('_delta_t')
        if '_delta_r' in pre_checkpoint['state_dict']:
            update_name.append('_delta_r')
        update_dict = dict([(key, pre_checkpoint['state_dict'][key]) for key in update_name])
        sugar_dict.update(update_dict)
        sugar.load_state_dict(sugar_dict)

    if pre_checkpoint and (not pre_color_fix):
        sugar.pre_sh_coordinates = torch.cat([pre_checkpoint['state_dict']['_sh_coordinates_dc'], pre_checkpoint['state_dict']['_sh_coordinates_rest']], dim=1)
        if args.tracking_data and pre_ckpt_update:
            pre_sh_mask = track_face_mask.repeat(6)
            sugar.pre_sh_coordinates = sugar.pre_sh_coordinates[pre_sh_mask]

    else:
        sh_reg_loss_from = 999999
        
    CONSOLE.print(f'\nSuGaR model has been initialized.')
    CONSOLE.print(sugar)
    CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in sugar.parameters() if p.requires_grad)}')
    CONSOLE.print(f'Checkpoints will be saved in {sugar_checkpoint_path}')
    
    CONSOLE.print("\nModel parameters:")
    for name, param in sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)
 
    torch.cuda.empty_cache()
    
    # Compute scene extent
    cameras_spatial_extent = sugar.get_cameras_spatial_extent()

    # ====================Initialize optimizer====================
    if use_custom_bbox:
        bbox_radius = ((torch.tensor(fg_bbox_max) - torch.tensor(fg_bbox_min)).norm(dim=-1) / 2.).item()
    else:
        bbox_radius = cameras_spatial_extent
    n_vertices_in_fg = len(o3d_mesh.triangles)
    spatial_lr_scale = 10. * bbox_radius / torch.tensor(n_vertices_in_fg).pow(1/2).item()  # ZCW change 10. *
    print("Using as spatial_lr_scale:", spatial_lr_scale, "with bbox_radius:", bbox_radius, "and n_vertices_in_fg:", n_vertices_in_fg)
    
    opt_params = OptimizationParams(
        iterations=num_iterations,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_lr_delay_mult=position_lr_delay_mult,
        position_lr_max_steps=position_lr_max_steps,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
    )
    optimizer = SuGaROptimizer(sugar, opt_params, spatial_lr_scale=spatial_lr_scale)
    CONSOLE.print("Optimizer initialized.")
    CONSOLE.print("Optimization parameters:")
    CONSOLE.print(opt_params)
    
    CONSOLE.print("Optimizable parameters:")
    for param_group in optimizer.optimizer.param_groups:
        CONSOLE.print(param_group['name'], param_group['lr'])
        
        
    # ====================Initialize densifier====================
    if use_densifier:
        gaussian_densifier = SuGaRDensifier(
            sugar_model=sugar,
            sugar_optimizer=optimizer,
            max_grad=densify_grad_threshold,
            min_opacity=prune_opacity_threshold,
            max_screen_size=densify_screen_size_threshold,
            scene_extent=cameras_spatial_extent,
            percent_dense=densification_percent_distinction,
            )
        CONSOLE.print("Densifier initialized.")
        
    
    # ====================Loss function====================
    if loss_function == 'l1':
        loss_fn = l1_loss
    elif loss_function == 'l2':
        loss_fn = l2_loss
    elif loss_function == 'l1+dssim':
        def loss_fn(pred_rgb, gt_rgb):
            return (1.0 - dssim_factor) * l1_loss(pred_rgb, gt_rgb) + dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))
    CONSOLE.print(f'Using loss function: {loss_function}')

    bg_color = [0.0, 1.0, 0.0]

    # ZCW save config
    cfg_dict = {"Source path": source_path,
                "Gaussian Splatting checkpoint path": gs_checkpoint_path,
                "SUGAR checkpoint path": sugar_checkpoint_path,
                "Surface mesh to bind to": surface_mesh_to_bind_path,
                "Surface mesh from MS": args.coarse_mesh_from_MS,
                "Iteration to load": iteration_to_load,
                "SH level": sh_levels,
                "Normal consistency enable": use_surface_mesh_normal_consistency_loss,
                "Normal consistency factor": surface_mesh_normal_consistency_factor,
                "Laplacian smoothing enable": use_surface_mesh_laplacian_smoothing_loss,
                "Laplacian smoothing factor": surface_mesh_laplacian_smoothing_factor,
                "Number of gaussians per surface triangle": n_gaussians_per_surface_triangle,
                "Number of vertices in the foreground": n_vertices_in_fg,
                "Use eval split": use_eval_split,
                "Export ply at the end": export_ply_at_the_end,
                "Spatial lr scale": spatial_lr_scale,
                "BG color": bg_color,
                "Use opacity reg loss": cfg.use_opacity_reg,
                "Min opacity": cfg.min_opacity,
                "Pre checkpoint": args.pre_checkpoint_path,
                "Pre checkpoint for color init": pre_ckpt_color_init,
                "Pre checkpoint for scale rot init": pre_ckpt_scale_R_init,
                "Pre checkpoint for opacity init": pre_ckpt_opacity_init,
                "Pre color fix": pre_color_fix,
                "Regularize": regularize,
                "Mask loss factor": cfg.mask_loss_factor,
                "Mask loss from": cfg.mask_loss_from,
                "Depth loss factor": cfg.depth_loss_factor,
                "Depth loss from": cfg.depth_loss_from,
                "SH reg loss factor": cfg.sh_reg_loss_factor,
                "SH reg loss from": sh_reg_loss_from,
                "Area reg loss factor": area_reg_loss_factor,
                "Area reg loss from": area_reg_loss_from,
                # "Area loss": "torch.relu(mean_area / face_area - 2).mean()",
                "Max scale for gaussian": max_gaussian_scale,
                "Max gaussian scalar": max_gaussian_scalar,
                "Min scale for gaussian": min_gaussian_scale,
                "Min gaussian scalar": min_gaussian_scalar,
                "SH warmup every": sh_warmup_every,
                "Loose bind from": loose_bind_from,
                "Loose bind factor t": cfg.loose_bind_factor_t,
                "Loose bind factor r": cfg.loose_bind_factor_r,
                "Mesh prop for detect": mesh_prop_for_detect,
                "Use HumanRF depth and mask": args.from_humanrf,
                }
    if args.ref_mesh_path or args.tracking_data:
        cfg_dict.update(
            {
                "Ref mesh path": args.ref_mesh_path,
                "Ref tracking data": args.tracking_data,
                "Ref edge iso loss factor": edge_iso_loss_factor,
                "Ref edge iso loss from": edge_iso_loss_from,
                "Ref area iso loss factor": area_iso_loss_factor,
                "Ref area iso loss from": area_iso_loss_from,
            })

    cfg_json = json.dumps(cfg_dict, sort_keys=True, indent=4, separators=(',', ': '))

    fout = open(os.path.join(sugar_checkpoint_path, 'config.json'), 'w')
    fout.write(cfg_json)
    fout.close()
    
    
    # ====================Start training====================
    sugar.train()
    epoch = 0
    iteration = 0
    train_losses = []
    t0 = time.time()
    
    for batch in range(9_999_999):
        if iteration >= num_iterations:
            break
        
        # Shuffle images
        shuffled_idx = torch.randperm(len(nerfmodel.training_cameras))
        train_num_images = len(shuffled_idx)
        
        # We iterate on images
        for i in range(0, train_num_images, train_num_images_per_batch):
            iteration += 1
            
            # Update learning rates
            optimizer.update_learning_rate(iteration)
            
            start_idx = i
            end_idx = min(i+train_num_images_per_batch, train_num_images)
            
            camera_indices = shuffled_idx[start_idx:end_idx]
            cmr_i = camera_indices.item()
            
            # Computing rgb predictions
            if not no_rendering:
                outputs = sugar.render_image_gaussian_rasterizer( 
                    camera_indices=cmr_i,
                    verbose=False,
                    bg_color=bg_color,
                    sh_deg=current_sh_levels-1,
                    sh_rotations=None,
                    compute_color_in_rasterizer=compute_color_in_rasterizer,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=use_densifier or regularize,
                    quaternions=None,
                    use_same_scale_in_all_directions=use_same_scale_in_all_directions,
                    return_opacities=False,
                    )
                if use_densifier or regularize:
                    pred_rgb = outputs['image'].view(-1, 
                        sugar.image_height_list[cmr_i],
                        sugar.image_width_list[cmr_i],
                        3)
                    if use_densifier or regularize:
                        radii = outputs['radii']
                        viewspace_points = outputs['viewspace_points']
                else:
                    pred_rgb = outputs.view(-1, sugar.image_height_list[cmr_i], sugar.image_width_list[cmr_i], 3)
                
                pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)  # TODO: Change for torch.permute
                
                # Gather rgb ground truth
                gt_image = nerfmodel.get_gt_image(camera_indices=cmr_i)
                gt_rgb = gt_image.view(-1, sugar.image_height_list[cmr_i], sugar.image_width_list[cmr_i], 3)
                gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)
                    
                # Compute loss
                if cfg.use_margin:  # For ActorHQ dataset, due to cx cy != shape / 2
                    margin = cmr_margin_list[cmr_i]
                    loss = loss_fn(pred_rgb[..., margin[2]:-margin[3], margin[0]:-margin[1]],
                                   gt_rgb[..., margin[2]:-margin[3], margin[0]:-margin[1]])

                    # tmp = pred_rgb[..., margin[2]:-margin[3], margin[0]:-margin[1]]
                    # tmp = tmp[0].transpose(-2, -3).transpose(-1, -2)
                    # tmp = tmp.detach().cpu().numpy()
                    # cv2.imwrite(sugar_checkpoint_path + f"debug/pred_{cmr_i:06d}.jpg", tmp * 255)
                else:
                    loss = loss_fn(pred_rgb, gt_rgb)

                loss_dict = {"rgb_loss": loss.detach().item()}

                # ZCW add depth related losses
                if (iteration > cfg.mask_loss_from) or (iteration > cfg.depth_loss_from):
                    depth_alpha = False

                    fov_camera = nerfmodel.training_cameras.p3d_cameras[cmr_i]
                    point_depth = fov_camera.get_world_to_view_transform().transform_points(sugar.points)[..., 2:]
                    if not depth_alpha:
                        point_depth = point_depth.expand(-1, 3)
                        max_depth = 10  # point_depth.max() * 2
                        pred_depth = sugar.render_image_gaussian_rasterizer(
                            camera_indices=cmr_i,
                            bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=sugar.device),
                            sh_deg=0,
                            compute_color_in_rasterizer=False,  # compute_color_in_rasterizer,
                            compute_covariance_in_rasterizer=True,
                            return_2d_radii=False,
                            use_same_scale_in_all_directions=False,
                            point_colors=point_depth,
                        )[..., 0]
                    else:
                        point_depth = point_depth.repeat(1, 3)
                        point_depth[..., 2] = 1
                        pred_depth_alpha = sugar.render_image_gaussian_rasterizer(
                            camera_indices=cmr_i,
                            bg_color=[0.0, 0.0, 0.0],
                            sh_deg=0,
                            compute_color_in_rasterizer=False,  # compute_color_in_rasterizer,
                            compute_covariance_in_rasterizer=True,
                            return_2d_radii=False,
                            use_same_scale_in_all_directions=False,
                            point_colors=point_depth,
                        )
                        pred_depth = pred_depth_alpha[..., 0]
                        pred_alpha = pred_depth_alpha[..., 2]
                        pred_depth = pred_depth / (pred_alpha + 1e-8)

                    gt_depth = nerfmodel.get_gt_depth(camera_indices=cmr_i)[..., 0]
                    if iteration > cfg.depth_loss_from:
                        fg_mask = (gt_depth < 10)
                        fg_pred_depth = pred_depth[fg_mask]
                        fg_gt_depth = gt_depth[fg_mask]
                        # if iteration > 5000:
                        #     depth_loss_factor = 1
                        depth_loss = cfg.depth_loss_factor * (fg_pred_depth - fg_gt_depth).abs().mean()  # L1
                        # loss = loss + depth_loss_factor * ((fg_pred_depth - fg_gt_depth) ** 2).mean()  # L2
                        loss = loss + depth_loss
                        loss_dict['depth_loss'] = depth_loss.detach().item()

                    if iteration > cfg.mask_loss_from:
                        # gt_mask = nerfmodel.get_gt_mask(camera_indices=cmr_i)[..., 0]
                        # bg_mask = (gt_mask < 0.1)
                        bg_mask = (gt_depth > 10)
                        if not depth_alpha:
                            bg_pred_depth = pred_depth[bg_mask]
                            mask_loss = cfg.mask_loss_factor * (bg_pred_depth - max_depth).abs().mean()  # L1
                            # loss = loss + mask_loss_factor * ((bg_pred_depth - max_depth) ** 2).mean()  # L2
                        else:
                            bg_pred_alpha = pred_alpha[bg_mask]
                            mask_loss = cfg.mask_loss_factor * bg_pred_alpha.mean() * 5  # L1
                            # fg_pred_alpha = pred_alpha[~bg_mask]
                            # mask_loss += mask_loss_factor * (1 - fg_pred_alpha).mean()
                        loss = loss + mask_loss
                        loss_dict['mask_loss'] = mask_loss.detach().item()

                # ZCW add SH Reg loss
                if iteration > sh_reg_loss_from:
                    # loss = loss + sh_reg_loss_factor * (sugar.pre_sh_coordinates - sugar.sh_coordinates).abs().mean()  # L1
                    # loss = loss + sh_reg_loss_factor * ((sugar.pre_sh_coordinates - sugar.sh_coordinates) ** 2).mean()  # L2
                    if args.tracking_data and pre_ckpt_update:
                        loss = loss + cfg.sh_reg_loss_factor * ((sugar.pre_sh_coordinates[:, 0, :] - sugar.sh_coordinates[:(track_face_num*6), 0, :]) ** 2).mean()  # L2
                    else:
                        loss = loss + cfg.sh_reg_loss_factor * ((sugar.pre_sh_coordinates[:, 0, :] - sugar.sh_coordinates[:, 0, :]) ** 2).mean()  # L2
                
                if regularize:
                    raise RuntimeError("SuGaR regularize is disabled in GauSTAR")
                                
            else:
                loss = 0.

            # Surface & mesh losses
            if use_surface_losses:
                surface_mesh = sugar.surface_mesh

                if use_surface_mesh_laplacian_smoothing_loss:
                    loss = loss + surface_mesh_laplacian_smoothing_factor * mesh_laplacian_smoothing(
                        surface_mesh, method=surface_mesh_laplacian_smoothing_method)

                if use_surface_mesh_normal_consistency_loss:
                    nc_loss = surface_mesh_normal_consistency_factor * mesh_normal_consistency(surface_mesh)
                    loss = loss + nc_loss
                    loss_dict['nc_loss'] = nc_loss.detach().item()

                if args.ref_mesh_path:
                    if iteration > edge_iso_loss_from:
                        verts_edges = surface_mesh.verts_packed()[surface_mesh.edges_packed()]
                        v0, v1 = verts_edges.unbind(1)
                        edge_len = (v0 - v1).norm(dim=1, p=2)
                        # loss = loss + edge_iso_loss_factor * (edge_len - ref_edge_len).abs().mean()
                        edge_loss = edge_iso_loss_factor * ((edge_len - ref_edge_len) ** 2).mean()
                        loss = loss + edge_loss
                        loss_dict['edge_loss'] = edge_loss.detach().item()

                    if iteration > area_iso_loss_from:
                        face_area = surface_mesh.faces_areas_packed()
                        area_loss = area_iso_loss_factor * (face_area - ref_area).abs().mean()
                        # area_diff = (face_area - ref_area) * 1e3
                        # area_loss = area_iso_loss_factor * (area_diff ** 2).mean()
                        loss = loss + area_loss
                        loss_dict['area_loss'] = area_loss.detach().item()

                    # gaussian_points = sugar.mesh_vert
                    # gaussian_points = Pointclouds(points=[gaussian_points])
                    # loss = loss + ref_mesh_loss_factor * point_mesh_face_distance(meshes=ref_mesh, pcls=gaussian_points)
                    # loss = loss + ref_mesh_loss_factor * chamfer_distance(meshes=ref_mesh, pcls=gaussian_points)

                if iteration > area_reg_loss_from:
                    face_area = surface_mesh.faces_areas_packed()
                    with torch.no_grad():
                        mean_area = face_area.mean()
                    area_reg_loss = torch.relu(mean_area / face_area - 2).mean()
                    loss = loss + area_reg_loss_factor * area_reg_loss

                if sugar.is_loose_bind() or iteration == loose_bind_from:
                    # Unbind Gaussians
                    if iteration == loose_bind_from:
                        with torch.no_grad():
                            unbind_weight = detect_topo_err(sugar, nerfmodel, sugar_checkpoint_path, cmr_npz, iteration,
                                                            use_depth_loss=True, depth_scalar=3,
                                                            use_color_loss=False, use_densifier_grad=False,
                                                            mesh_prop=mesh_prop_for_detect,
                                                            save_inter=False, save_render=False, save_mesh=True)
                        unbind_weight = torch.tensor(1 - unbind_weight.repeat(sugar.n_gaussians_per_surface_triangle))
                        topo_change_face_num = (unbind_weight == 0).sum()
                        if topo_change_face_num < 100:
                            print(f"topo_change_face_num: {topo_change_face_num}, skip unbinding!")
                            # sugar.rebind()
                        else:
                            print(f"topo_change_face_num: {topo_change_face_num}, unbind!")
                            sugar.loose_bind()
                            sugar.unbind_loss_weight = unbind_weight[:, None].expand(-1, 3).to(sugar.device)
                    if sugar.is_loose_bind():
                        loss = loss + cfg.loose_bind_factor_t * (sugar.unbind_loss_weight * sugar.delta_t.abs()).mean()
                        loss = loss + cfg.loose_bind_factor_r * (sugar.unbind_loss_weight * sugar.delta_r[..., 1:].abs()).mean()

                # ZCW add opacity_reg loss
                if cfg.use_opacity_reg:
                    splat_opacities = sugar.strengths.view(-1, 1)
                    if sugar.is_loose_bind() and 0:  # disabled
                        loss = loss + (sugar.unbind_loss_weight[:, :1] * torch.relu(cfg.min_opacity - splat_opacities)).mean()
                    else:
                        loss = loss + torch.relu(cfg.min_opacity - splat_opacities).mean()

            # Update parameters
            loss.backward()
            
            # Densification
            with torch.no_grad():
                if (not no_rendering) and use_densifier and (iteration < densify_until_iter):
                    gaussian_densifier.update_densification_stats(viewspace_points, radii, visibility_filter=radii>0)

                    if iteration > densify_from_iter and (iteration+1) % densification_interval == 0:

                        if densifier_detection_only:
                            densifier_mask, densifier_grad = gaussian_densifier.densify_detection(densify_grad_threshold, return_grad=True)

                            # densifier_mask = densifier_mask.detach().cpu().numpy()
                            # densifier_face_mask = densifier_mask.reshape((-1, n_gaussians_per_surface_triangle))
                            # densifier_face_mask = densifier_face_mask.sum(axis=-1)

                            densifier_grad = densifier_grad.detach().cpu().numpy()
                            densifier_face_grad = densifier_grad.reshape((-1, n_gaussians_per_surface_triangle))
                            densifier_face_grad = np.average(densifier_face_grad, axis=-1)

                            if (loose_bind_from - densification_interval) < (iteration+1) <= loose_bind_from:
                                vert = sugar.surface_mesh.verts_list()[0].detach().cpu().numpy()
                                face = sugar.surface_mesh.faces_list()[0].detach().cpu().numpy()

                                face_color = densifier_face_grad / 0.0001 * 255
                                face_color = face_color[..., None].repeat(3, axis=1)

                                face_color = np.clip(face_color, 0, 255)
                                obj_mesh = trimesh.Trimesh(vertices=vert, faces=face, face_colors=face_color,
                                                           process=False)
                                os.makedirs(sugar_checkpoint_path + 'detect/', exist_ok=True)
                                obj_mesh.export(sugar_checkpoint_path + f'detect/densifier_grad_{(iteration+1):04d}.obj')
                        else:
                            size_threshold = gaussian_densifier.max_screen_size if iteration > opacity_reset_interval else None
                            gaussian_densifier.densify_and_prune(densify_grad_threshold, prune_opacity_threshold,
                                                        cameras_spatial_extent, size_threshold)
                            CONSOLE.print("Gaussians densified and pruned. New number of gaussians:", len(sugar.points))

                    if (not densifier_detection_only) and iteration % opacity_reset_interval == 0:
                        gaussian_densifier.reset_opacity()
                        CONSOLE.print("Opacity reset.")
            
            # Optimization step
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            
            # Print loss
            if cfg.print_training_info and (iteration == 1 or iteration % print_loss_every_n_iterations == 0):
                CONSOLE.print(f'\n-------------------\nIteration: {iteration}')
                train_losses.append(loss.detach().item())
                CONSOLE.print(f"loss: {loss:>7f}  [{iteration:>5d}/{num_iterations:>5d}]",
                    "computed in", (time.time() - t0) / 60., "minutes.")
                # CONSOLE.print(f"[{iteration:>5d}/{num_iterations:>5d}]",
                #     "computed in", (time.time() - t0) / 60., "minutes.")
                for keys, values in loss_dict.items():
                    CONSOLE.print(keys, values)
                with torch.no_grad():
                    # scales = sugar.scaling.detach()
             
                    CONSOLE.print("------Stats-----")
                    CONSOLE.print("---Min, Max, Mean, Std")
                    CONSOLE.print("Points:", sugar.points.min().item(), sugar.points.max().item(), sugar.points.mean().item(), sugar.points.std().item(), sep='   ')
                    CONSOLE.print("Scaling factors:", sugar.scaling[..., 1:].min().item(), sugar.scaling[..., 1:].max().item(), sugar.scaling[..., 1:].mean().item(), sugar.scaling[..., 1:].std().item(), sep='   ')
                    CONSOLE.print("Quaternions:", sugar.quaternions.min().item(), sugar.quaternions.max().item(), sugar.quaternions.mean().item(), sugar.quaternions.std().item(), sep='   ')
                    CONSOLE.print("Sh coordinates dc:", sugar._sh_coordinates_dc.min().item(), sugar._sh_coordinates_dc.max().item(), sugar._sh_coordinates_dc.mean().item(), sugar._sh_coordinates_dc.std().item(), sep='   ')
                    CONSOLE.print("Sh coordinates rest:", sugar._sh_coordinates_rest.min().item(), sugar._sh_coordinates_rest.max().item(), sugar._sh_coordinates_rest.mean().item(), sugar._sh_coordinates_rest.std().item(), sep='   ')
                    CONSOLE.print("Opacities:", sugar.strengths.min().item(), sugar.strengths.max().item(), sugar.strengths.mean().item(), sugar.strengths.std().item(), sep='   ')
                t0 = time.time()
                
            # Save model
            if (iteration % save_model_every_n_iterations == 0) or (iteration in save_milestones):
                CONSOLE.print("Saving model...")
                model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
                sugar.save_model(path=model_path,
                                train_losses=train_losses,
                                epoch=epoch,
                                iteration=iteration,
                                optimizer_state_dict=optimizer.state_dict(),
                                )
                # if optimize_triangles and iteration >= optimize_triangles_from:
                #     rm.save_model(os.path.join(rc_checkpoint_path, f'rm_{iteration}.pt'))
                CONSOLE.print("Model saved.")
            
            if iteration >= num_iterations:
                break
            
            if do_sh_warmup and (iteration > 0) and (current_sh_levels < sh_levels) and (iteration % sh_warmup_every == 0):
                current_sh_levels += 1
                CONSOLE.print("Increasing number of spherical harmonics levels to", current_sh_levels)
        
        epoch += 1

    CONSOLE.print(f"Training finished after {num_iterations} iterations with loss={loss.detach().item()}.")
    CONSOLE.print("Saving final model...")
    model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
    sugar.save_model(path=model_path,
                    train_losses=train_losses,
                    epoch=epoch,
                    iteration=iteration,
                    # optimizer_state_dict=optimizer.state_dict(),
                    )

    CONSOLE.print("Final model saved.")
    
    if export_ply_at_the_end:
        CONSOLE.print("\nExporting ply file with refined Gaussians...")
        tmp_list = model_path.split(os.sep)

        refined_ply_save_path = os.path.join(sugar_checkpoint_path, tmp_list[-2]) + ".ply"

        # Export and save ply
        refined_gaussians = convert_refined_sugar_into_gaussians(sugar)
        refined_gaussians.save_ply(refined_ply_save_path)
        CONSOLE.print("Ply file exported. This file is needed for using the dedicated viewer.")
    
    return model_path, sugar.is_loose_bind()

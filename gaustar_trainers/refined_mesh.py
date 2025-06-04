import os
import open3d as o3d
import torch
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points
from pytorch3d.io import save_obj
from gaustar_scene.gs_model import GaussianSplattingWrapper
from gaustar_scene.sugar_model import SuGaR
from gaustar_utils.spherical_harmonics import SH2RGB

from rich.console import Console
import trimesh

import cv2
import numpy as np
from tqdm import tqdm
import json

from gaustar_tools.warp_mesh import get_depth_edge, project, query_at_image
from gaustar_tools.warp_mesh import mesh_vert_propagate, build_voxel_from_pc, interpolate_in_voxel
from pytorch3d.renderer.cameras import look_at_view_transform

max_depth = 10


def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins = (viewpoint_cam.projection_matrix @ ndc2pix)[:3, :3].T
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx=intrins[0, 2].item(),
            cy=intrins[1, 2].item(),
            fx=intrins[0, 0].item(),
            fy=intrins[1, 1].item()
        )

        extrinsic = np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


def sample_cam(intr, dist=3, look_at_y=1.2, flip_xy=False):
    cam_list = []
    at = np.array([[0, look_at_y, 0]])
    dist_vec = np.array([0, 0, dist])
    for azim in range(0, 360, 30):
        for elev in range(-40, 41, 20):

            R, t = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=at, degrees=True, up=[[0, -1, 0]])
            R = R[0].numpy()

            t = dist_vec - np.matmul(R, at[0])
            # t = t[0].numpy()
            # at_np = np.array(at)[0]
            # check = np.matmul(R, at_np) + t
            # print(check)

            if flip_xy:
                R[:, :2] *= -1
                t[:2] *= -1
            extr = np.identity(4)
            extr[:3, :3] = R
            extr[:3, 3] = t
            camera = o3d.camera.PinholeCameraParameters()
            camera.extrinsic = extr
            camera.intrinsic = intr
            cam_list.append(camera)
    return cam_list


def find_boundary_verts(mesh: trimesh.Trimesh, pc_aabb=None, cut_inner=False, pad=0.02):
    unique_edges = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    unique_edges = mesh.edges[unique_edges]
    boundary_vert_idx = np.unique(unique_edges.flatten())

    if pc_aabb is None:
        return boundary_vert_idx

    boundary_verts = mesh.vertices[boundary_vert_idx]

    if cut_inner:  # return vert in a larger size of aabb
        new_aabb = np.zeros_like(pc_aabb)
        new_aabb[0] = pc_aabb[0] - pad
        new_aabb[1] = pc_aabb[1] + pad
        inbox_vert_mask = find_points_in_boundingbox(boundary_verts, new_aabb)
        return boundary_vert_idx[inbox_vert_mask]

    else:  # return vert located in boundary faces that across the boundary
        inside_vert_mask = find_points_in_boundingbox(mesh.vertices, pc_aabb)
        inside_vert_ids = np.where(inside_vert_mask)[0]
        inside_face_vert_mask = np.isin(mesh.faces, inside_vert_ids)
        inside_face_mask = np.any(inside_face_vert_mask, axis=1)
        full_inside_face_mask = np.all(inside_face_vert_mask, axis=1)
        boundary_face_mask = inside_face_mask & (~full_inside_face_mask)

        vert_on_boundary_face = np.unique(mesh.faces[boundary_face_mask].flatten())
        vert_on_boundary_mask2 = np.isin(boundary_vert_idx, vert_on_boundary_face)
        return boundary_vert_idx[vert_on_boundary_mask2]


def reset_duplicate_vert(vert, face, possible_vid, debug_check=0):
    vert_group_list = trimesh.grouping.group_rows(vert[possible_vid])
    for vert_group in vert_group_list:
        new_vert_gi = min(vert_group)
        if debug_check:
            assert new_vert_gi < debug_check
        new_vert_idx = possible_vid[new_vert_gi]
        for old_vert_gi in vert_group:
            old_vert_idx = possible_vid[old_vert_gi]
            face[face == old_vert_idx] = new_vert_idx


def merge_vert_around_holes(mesh: trimesh.Trimesh, max_hole_vert_num=10):
    # find edge and vert in the hole
    # unique_edges = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    regular_edges_id = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=2)
    hole_edge_mask = np.ones(mesh.edges_sorted.shape[0], dtype=bool)
    hole_edge_mask[regular_edges_id.flatten()] = False
    hole_edges = mesh.edges[hole_edge_mask]
    hole_verts = np.unique(hole_edges.flatten())

    if hole_verts.size == 0:
        return

    # reset vert idx in edge
    for i, vert in enumerate(hole_verts):
        hole_edges[hole_edges == vert] = i

    vert_hole_label = trimesh.graph.connected_component_labels(hole_edges)

    hole_num = vert_hole_label.max() + 1
    for hole_idx in range(hole_num):
        vert_gi = np.where(vert_hole_label == hole_idx)
        vert_idx = hole_verts[vert_gi]
        if vert_idx.shape[0] > max_hole_vert_num:
            continue
        new_vert_idx = vert_idx.min()
        # this_hole_vert_ids = hole_verts[vert_hole_label==hole_idx]
        mesh.vertices[vert_idx] = mesh.vertices[new_vert_idx]

    reset_duplicate_vert(mesh.vertices, mesh.faces, hole_verts)
    return


def connect_two_meshes(mesh1: trimesh.Trimesh, boundary_vid1, mesh2: trimesh.Trimesh, boundary_vid2):

    vert_num1 = mesh1.vertices.shape[0]
    boundary_num1 = boundary_vid1.shape[0]

    # Step: vert in mesh 2 move towards mesh 1
    pc1 = mesh1.vertices[boundary_vid1]
    pc2 = mesh2.vertices[boundary_vid2]
    pc_neighbor_2bto1b = knn_points(torch.from_numpy(pc2[None]).float(), torch.from_numpy(pc1[None]).float())
    pc2bto1b_idx = pc_neighbor_2bto1b.idx.numpy().squeeze()
    pc2bto1b_max_dist2 = pc_neighbor_2bto1b.dists.max().numpy()
    # vert2_ori = mesh2.vertices[boundary_vid2].copy()
    # pc2bto1b_max_dist2 = np.linalg.norm((vert2_ori - pc1[pc2bto1b_idx]), axis=1).max()
    mesh2.vertices[boundary_vid2] = pc1[pc2bto1b_idx]

    # Step: vert in mesh 1 move towards mesh 2
    pc2 = mesh2.vertices[boundary_vid2]
    pc_neighbor_1bto2b = knn_points(torch.from_numpy(pc1[None]).float(), torch.from_numpy(pc2[None]).float())
    pc1bto2b_idx = pc_neighbor_1bto2b.idx.numpy().squeeze()
    pc1bto2b_max_dist2 = pc_neighbor_1bto2b.dists.max().numpy()
    mesh1.vertices[boundary_vid1] = pc2[pc1bto2b_idx]

    # Step: build new mesh
    vert = np.concatenate((mesh1.vertices, mesh2.vertices))
    face = np.concatenate((mesh1.faces, mesh2.faces + vert_num1))
    face_color = np.concatenate((mesh1.visual.face_colors, mesh2.visual.face_colors))
    boundary_vids = np.concatenate((boundary_vid1, boundary_vid2 + vert_num1))

    # Step: remove duplicate verts
    # boundary_verts = np.concatenate((mesh1.vertices[boundary_vert1], mesh1.vertices[boundary_vert1]))
    reset_duplicate_vert(vert, face, boundary_vids, debug_check=boundary_num1)

    connected_mesh = trimesh.Trimesh(vertices=vert, faces=face, face_colors=face_color, process=False)
    assert connected_mesh.faces.shape[0] == mesh1.faces.shape[0] + mesh2.faces.shape[0]

    valid_face_mask1 = connected_mesh.nondegenerate_faces()
    connected_mesh.update_faces(valid_face_mask1)
    connected_mesh.remove_unreferenced_vertices()

    # Step: repair holes
    merge_vert_around_holes(connected_mesh)
    # connected_mesh.fill_holes()

    valid_face_mask2 = connected_mesh.nondegenerate_faces()
    connected_mesh.update_faces(valid_face_mask2)
    connected_mesh.remove_unreferenced_vertices()

    valid_face_mask = valid_face_mask1
    valid_face_mask[valid_face_mask1] = valid_face_mask2
    # assert connected_mesh.is_watertight
    # if not connected_mesh.is_watertight:
    #     merge_vert_around_holes(connected_mesh)

    max_dist = np.sqrt(np.maximum(pc2bto1b_max_dist2, pc1bto2b_max_dist2))

    return {"connected_mesh": connected_mesh,
            "valid_face_mask": valid_face_mask,
            "max_dist": max_dist}


def find_points_in_boundingbox(vertices, bb):
    min_v = bb[0]
    max_v = bb[1]
    vert_mask = (vertices[:, 0] > min_v[0]) & (vertices[:, 0] < max_v[0]) & \
                (vertices[:, 1] > min_v[1]) & (vertices[:, 1] < max_v[1]) & \
                (vertices[:, 2] > min_v[2]) & (vertices[:, 2] < max_v[2])
    return vert_mask


def cut_mesh_by_boundingbox(mesh: trimesh.Trimesh, bb, cut_inner=False, inplace=False):
    vert_mask = find_points_in_boundingbox(mesh.vertices, bb)
    inside_indices = np.where(vert_mask)[0]
    inside_face_vert_mask = np.isin(mesh.faces, inside_indices)
    inside_face_mask = np.any(inside_face_vert_mask, axis=1)
    full_inside_face_mask = np.all(inside_face_vert_mask, axis=1)
    boundary_face_mask = inside_face_mask & (~full_inside_face_mask)

    out_dir = {}
    if cut_inner:
        inside_face_mask = ~inside_face_mask
    # else:
    #     out_dir['boundary_face_mask'] = boundary_face_mask[inside_face_mask]
    out_dir['inside_face_mask'] = inside_face_mask

    if inplace:
        cut_mesh = mesh
    else:
        cut_mesh = mesh.copy()
    cut_mesh.update_faces(inside_face_mask)
    cut_mesh.remove_unreferenced_vertices()
    # remove_unused_vert(cut_mesh)
    out_dir['cut_mesh'] = cut_mesh

    return out_dir


def combine_overlap_aabbs(aabb_list):
    new_list = []
    for aabb in aabb_list:
        aabb_points = np.zeros((8, 3))
        aabb_points[0] = np.array([aabb[0, 0], aabb[0, 1], aabb[0, 2]])
        aabb_points[1] = np.array([aabb[0, 0], aabb[0, 1], aabb[1, 2]])
        aabb_points[2] = np.array([aabb[0, 0], aabb[1, 1], aabb[0, 2]])
        aabb_points[3] = np.array([aabb[0, 0], aabb[1, 1], aabb[1, 2]])
        aabb_points[4] = np.array([aabb[1, 0], aabb[0, 1], aabb[0, 2]])
        aabb_points[5] = np.array([aabb[1, 0], aabb[0, 1], aabb[1, 2]])
        aabb_points[6] = np.array([aabb[1, 0], aabb[1, 1], aabb[0, 2]])
        aabb_points[7] = np.array([aabb[1, 0], aabb[1, 1], aabb[1, 2]])

        overlap_id = -1
        for i in range(len(new_list)):
            aabb_points_overlap_flag = find_points_in_boundingbox(aabb_points, aabb_list[i])
            if np.any(aabb_points_overlap_flag):
                overlap_id = i
                break

        if overlap_id == -1:
            new_list.append(aabb)
            continue

        else:
            overlap_aabb = new_list[overlap_id]
            new_aabb = np.zeros_like(overlap_aabb)
            new_aabb[0] = np.minimum(overlap_aabb[0], aabb[0])
            new_aabb[1] = np.maximum(overlap_aabb[1], aabb[1])
            new_list[overlap_id] = new_aabb

    if len(new_list) == len(aabb_list):
        return new_list
    else:
        return combine_overlap_aabbs(new_list)


def get_outlier_cc_mask(face_adjacency, face_num_threshold=None, node_count=None):
    # RETURN False for outlier, True for other
    cc_label_for_face = trimesh.graph.connected_component_labels(face_adjacency, node_count=node_count)
    cc_face_count = np.bincount(cc_label_for_face)
    # cc_num = cc_label_for_face.max() + 1
    # cc_num = cc_face_count.size

    if face_num_threshold is None:
        # face_num_threshold = np.average(cc_face_count)
        face_num_threshold = cc_face_count.max() * 0.3
    else:
        face_num_threshold = np.minimum(face_num_threshold, cc_face_count.max() * 0.3)
    cc_return_label = np.where(cc_face_count >= face_num_threshold)[0]

    # cc_update_label = np.arange(cc_num)[cc_face_count > 15]
    face_return_mask = np.isin(cc_label_for_face, cc_return_label)
    return face_return_mask


@torch.no_grad()
def extract_mesh_fusion(refined_sugar: SuGaR, nerfmodel: GaussianSplattingWrapper,
                        voxel_size=0.008, sdf_trunc=0.02, depth_trunc=6, simplify_face_num=0,
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

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # flip_cmr_y = True
    default_intr_id = 0
    cmr_list = to_cam_open3d(nerfmodel.cam_list)

    cmr_sample_list = sample_cam(intr=cmr_list[default_intr_id].intrinsic)
    sample_num = len(cmr_sample_list)
    print("Camera sample number: ", sample_num)
    cmr_final_list = cmr_sample_list + cmr_list

    i = 0
    for cam_o3d in tqdm(cmr_final_list, desc="TSDF integration progress"):
        # default_intr_id = i+5
        if i < sample_num:
            rgb = refined_sugar.render_image_gaussian_rasterizer(
                camera_indices=default_intr_id,
                overwrite_extr=cam_o3d.extrinsic,
                bg_color=[0.0, 1.0, 0.0],
                sh_deg=nerfmodel.gaussians.active_sh_degree,
                compute_color_in_rasterizer=True,
                compute_covariance_in_rasterizer=True,
                return_2d_radii=False,
                use_same_scale_in_all_directions=False,
            ).clamp(min=0., max=1.).contiguous()
            label = "s"
        else:
            rgb = refined_sugar.render_image_gaussian_rasterizer(
                camera_indices=(i-sample_num),
                bg_color=[0.0, 1.0, 0.0],
                sh_deg=nerfmodel.gaussians.active_sh_degree,
                compute_color_in_rasterizer=True,
                compute_covariance_in_rasterizer=True,
                return_2d_radii=False,
                use_same_scale_in_all_directions=False,
            ).clamp(min=0., max=1.).contiguous()
            label = ""
        rgb = rgb.cpu().numpy()

        if i < sample_num:
            fov_camera = nerfmodel.training_cameras.p3d_cameras[default_intr_id]
            overwrite_extr_tensor = torch.Tensor(cam_o3d.extrinsic.copy()).to(refined_sugar.device)
            fov_camera.R = overwrite_extr_tensor[None, :3, :3].inverse()
            fov_camera.R[:, :, :2] *= -1
            fov_camera.T = overwrite_extr_tensor[None, :3, 3]
            fov_camera.T[:, :2] *= -1
        else:
            fov_camera = nerfmodel.training_cameras.p3d_cameras[(i-sample_num)]
        point_depth = fov_camera.get_world_to_view_transform().transform_points(refined_sugar.points)[..., 2:]
        point_depth = point_depth.repeat(1, 3)
        point_depth[..., 2] = 1
        if i < sample_num:
            depth_alpha = refined_sugar.render_image_gaussian_rasterizer(
                camera_indices=default_intr_id,
                overwrite_extr=cam_o3d.extrinsic,
                bg_color=[0.0, 0.0, 0.0],
                sh_deg=0,
                compute_color_in_rasterizer=False,
                compute_covariance_in_rasterizer=True,
                use_solid_surface=False,  # !!!
                use_same_scale_in_all_directions=False,
                point_colors=point_depth,
            ).contiguous()
        else:
            depth_alpha = refined_sugar.render_image_gaussian_rasterizer(
                camera_indices=(i-sample_num),
                # bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=refined_sugar.device),
                bg_color=[0.0, 0.0, 0.0],
                sh_deg=0,
                compute_color_in_rasterizer=False,
                compute_covariance_in_rasterizer=True,
                use_solid_surface=False,  # !!!
                use_same_scale_in_all_directions=False,
                point_colors=point_depth,
            ).contiguous()

        depth = depth_alpha[..., :1].cpu().numpy()
        alpha = depth_alpha[..., -1:].cpu().numpy()

        depth = depth / (alpha + 1e-8)

        # depth = nerfmodel.get_gt_depth(i).cpu().numpy()

        # if we have mask provided, use it
        if mask_backgrond:
            # bg_mask = depth > 0.7 * max_depth
            bg_mask = alpha < 0.5
            depth[bg_mask] = 0

        if remove_depth_edge:
            edge_depth = get_depth_edge(depth, ker_size=3)
            edge_vis = np.minimum(edge_depth / edge_depth.max() * 1000, 1)
            if save_dir:
                cv2.imwrite(save_dir + f"edge_vis_{label}{i:04d}.jpg", edge_vis * 255.0)
            edge_mask = edge_vis > 0.5
            depth[edge_mask] = 0

        if save_dir:
            cv2.imwrite(save_dir + f"color_{label}{i:06d}.jpg", rgb[..., ::-1] * 255)
            render_depth_vis = np.uint8(depth / max_depth * 255.0)
            render_depth_vis = cv2.applyColorMap(render_depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(save_dir + f"depth_{label}{i:06d}.jpg", render_depth_vis)

        # make open3d rgbd
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.asarray(rgb * 255, order="C", dtype=np.uint8)),
            o3d.geometry.Image(np.asarray(depth, order="C")),
            depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )

        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)
        i += 1

    mesh = volume.extract_triangle_mesh()
    if smooth:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
    if simplify_face_num > 0:
        # mesh_simplifier = pyfqmr.Simplify()
        # mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
        # mesh_simplifier.simplify_mesh(target_count=40000)
        # vertices, faces, _ = mesh_simplifier.getMesh()
        mesh = mesh.simplify_quadric_decimation(simplify_face_num)
    return mesh


@torch.no_grad()
def update_mesh_topo(refined_sugar: SuGaR, fusion_mesh: trimesh.Trimesh,
                     mesh_output_dir=None, log_out_dir=None, detected_mesh=None,
                     highlight_boundary=False, use_opacity=True, force_watertight=True, force_short_edge=False,
                     delta_threshold=0.6, cc_face_threshold=80, outlier_face_threshold=50, aabb_pad=0.02
                     # delta_threshold=0.6, cc_face_threshold=80, outlier_face_threshold=50, aabb_pad=0.02
                     ):

    log_dict = {
        "delta_threshold": delta_threshold,
        "cc_face_threshold": cc_face_threshold,
        "aabb_pad": aabb_pad,
        "use_detected_mesh": (detected_mesh is None),
        "force_watertight": force_watertight,
        "force_short_edge": force_short_edge,
        "outlier_face_threshold": outlier_face_threshold,
    }

    vert, face, face_color = refined_sugar.get_color_mesh()
    base_mesh = trimesh.Trimesh(vertices=vert, faces=face, face_colors=face_color, process=False)
    base_mesh_ori = base_mesh.copy()

    base_edge_vert = base_mesh.vertices[base_mesh.edges_unique]
    base_edge_average_len = np.linalg.norm((base_edge_vert[:, 0, :] - base_edge_vert[:, 1, :]), axis=1).mean()

    gs_pc = refined_sugar.points.detach().cpu().numpy()

    if mesh_output_dir is not None:
        merge_output_dir = mesh_output_dir + f"merge/pad{int(aabb_pad * 1000):d}mm/"
        os.makedirs(merge_output_dir, exist_ok=True)

    if detected_mesh is not None:
        # face_delta_mesh = trimesh.load_mesh(mesh_output_dir + "detect/mesh_001000/depth_diff_voxel_fc.obj")
        face_delta_mesh = detected_mesh.copy()
    else:
        # Compute face delta
        face_delta = refined_sugar.get_face_delta()
        face_delta = torch.clamp(face_delta * 100, max=1) * 255
        face_delta_np = face_delta.expand(-1, 3).cpu().numpy()
        face_delta_mesh = trimesh.Trimesh(vertices=vert, faces=face, face_colors=face_delta_np, process=False)
        if mesh_output_dir is not None:
            face_delta_mesh.export(mesh_output_dir + f"face_delta_full.obj")

        if use_opacity:
            face_opacity = refined_sugar.strengths.reshape(-1, refined_sugar.n_gaussians_per_surface_triangle)
            face_opacity = face_opacity.mean(axis=-1, keepdim=True).cpu().numpy()
            face_opacity_loss = np.maximum(0.8 - face_opacity, 0) * 100
            # face_opacity_loss = np.minimum(face_opacity_loss, 1)
            face_opacity_loss = face_opacity_loss.repeat(3, axis=-1) * 255
            face_delta_np = np.minimum(face_delta_np + face_opacity_loss, 255)
            face_delta_mesh = trimesh.Trimesh(vertices=vert, faces=face, face_colors=face_delta_np, process=False)
            if mesh_output_dir is not None:
                face_delta_mesh.export(mesh_output_dir + f"face_delta_with_opacity.obj")

    face_update_mask1 = (face_delta_mesh.visual.face_colors[..., 0] >= 255 * delta_threshold)
    face_update_idx = np.where(face_update_mask1)[0]
    face_delta_mesh.update_faces(face_update_mask1)
    face_delta_mesh.remove_unreferenced_vertices()
    if mesh_output_dir is not None:
        face_delta_mesh.export(mesh_output_dir + f"face_delta.obj")

    # Find cc (Connected Component) to be updated
    cc_label_for_face = trimesh.graph.connected_component_labels(face_delta_mesh.face_adjacency, node_count=face_delta_mesh.faces.shape[0])
    cc_face_count = np.bincount(cc_label_for_face)
    cc_update_label = np.where(cc_face_count > cc_face_threshold)[0]
    cc_update_num = cc_update_label.size
    if cc_update_num == 0:
        return {"cc_update_num": -1}
    face_update_mask2 = np.isin(cc_label_for_face, cc_update_label)
    # face_update_idx_cc = face_update_idx[face_update_mask2]

    face_delta_mesh.update_faces(face_update_mask2)
    face_delta_mesh.remove_unreferenced_vertices()
    if mesh_output_dir is not None:
        face_delta_mesh.export(mesh_output_dir + f"face_delta_select.obj")

    gs_pc = gs_pc.reshape(-1, refined_sugar.n_gaussians_per_surface_triangle, 3)
    gs_cc_pc = gs_pc[face_update_mask1][face_update_mask2]
    gs_cc_pc_ply = trimesh.points.PointCloud(gs_cc_pc.reshape(-1, 3))
    if mesh_output_dir is not None:
        gs_cc_pc_ply.export(mesh_output_dir + f"gs_cc_pc.ply")

    # Start updating
    cc_label_for_face = cc_label_for_face[face_update_mask2]
    track_face_mask = np.ones((face.shape[0]), dtype=bool)
    # base_mesh_mask = np.ones((face.shape[0]), dtype=bool)
    cc_success_flag = np.ones(cc_update_num, dtype=bool)

    pc_aabb_list = []
    # for cc_label in cc_update_label:
    for cc_idx in range(cc_update_num):
        cc_label = cc_update_label[cc_idx]

        face_in_this_cc_mask = (cc_label_for_face == cc_label)
        face_mesh_this_cc = face_delta_mesh.copy()
        face_mesh_this_cc.update_faces(face_in_this_cc_mask)
        face_mesh_this_cc.remove_unreferenced_vertices()
        # face_mesh_this_cc.export(mesh_output_dir + f"face_delta_cc_{cc_label}.obj")

        pc_this_cc_from_mesh = face_mesh_this_cc.vertices
        pc_this_cc_from_gs = gs_cc_pc[face_in_this_cc_mask].reshape(-1, 3)
        pc_this_cc = np.concatenate((pc_this_cc_from_mesh, pc_this_cc_from_gs), axis=0)
        pc_this_cc_ply = trimesh.points.PointCloud(pc_this_cc)
        # pc_this_cc_ply.export(mesh_output_dir + f"cc_pc_{cc_label}.ply")

        pc_aabb = pc_this_cc_ply.bounds
        pc_aabb[0] -= aabb_pad
        pc_aabb[1] += aabb_pad

        pc_aabb_list.append(pc_aabb)
        # face_idx = face_update_idx_cc[face_in_cc_mask]

    pc_aabb_list = combine_overlap_aabbs(pc_aabb_list)

    # for cc_idx in range(cc_update_num):
    max_dist_in_connection = 0
    for cc_idx in range(len(pc_aabb_list)):
        # cc_label = cc_update_label[cc_idx]
        cc_label = cc_idx
        pc_aabb = pc_aabb_list[cc_idx]

        cut_fusion_output = cut_mesh_by_boundingbox(fusion_mesh, pc_aabb, cut_inner=False)
        cut_fusion_mesh = cut_fusion_output['cut_mesh']
        # cut_fusion_boundary_face_mask = cut_fusion_output['boundary_face_mask']
        if cut_fusion_mesh.vertices.shape[0] == 0:
            cc_success_flag[cc_idx] = 0
            continue
        cut_fusion_mesh.fill_holes()
        outlier_mask = get_outlier_cc_mask(cut_fusion_mesh.face_adjacency,
                                           face_num_threshold=outlier_face_threshold,
                                           node_count=cut_fusion_mesh.faces.shape[0])
        assert outlier_mask.shape[0] == cut_fusion_mesh.faces.shape[0]
        # if outlier_mask.shape[0] != cut_fusion_mesh.faces.shape[0]:
        #     print("outlier_mask.shape[0]", outlier_mask.shape[0])
        #     print("cut_fusion_mesh.faces.shape[0]", cut_fusion_mesh.faces.shape[0])
        #     print("cut_fusion_mesh.face_adjacency.shape[0]", cut_fusion_mesh.face_adjacency.shape[0])
        cut_fusion_mesh.update_faces(outlier_mask)
        cut_fusion_mesh.remove_unreferenced_vertices()
        cut_fusion_mesh_boundary_verts = find_boundary_verts(cut_fusion_mesh, pc_aabb=pc_aabb, cut_inner=False)
        if cut_fusion_mesh_boundary_verts.shape[0] == 0:
            cc_success_flag[cc_idx] = 0
            continue
        if highlight_boundary:
            cut_fusion_mesh.visual.vertex_colors[cut_fusion_mesh_boundary_verts] = 255
        if mesh_output_dir is not None:
            cut_fusion_mesh.export(merge_output_dir + f"cut_fusion_mesh_{cc_label}.obj")

        cut_base_output = cut_mesh_by_boundingbox(base_mesh, pc_aabb, cut_inner=True, inplace=False)
        cut_base_mesh = cut_base_output['cut_mesh']
        if cut_base_mesh.vertices.shape[0] == 0:
            cc_success_flag[cc_idx] = 0
            continue
        cut_base_mesh_face_mask = cut_base_output['inside_face_mask']
        cur_base_mesh_face_num = cut_base_mesh.faces.shape[0]
        debug = cut_base_mesh.faces.copy()
        cut_base_mesh.fill_holes()
        assert (debug == cut_base_mesh.faces[:cur_base_mesh_face_num]).all()
        cut_base_mesh_boundary_verts = find_boundary_verts(cut_base_mesh, pc_aabb=pc_aabb, cut_inner=True)
        if cut_base_mesh_boundary_verts.shape[0] == 0:
            cc_success_flag[cc_idx] = 0
            continue
        if highlight_boundary:
            cut_base_mesh.visual.vertex_colors[cut_base_mesh_boundary_verts] = 255
        if mesh_output_dir is not None:
            cut_base_mesh.export(merge_output_dir + f"cut_base_mesh_{cc_label}.obj")

        connect_output = connect_two_meshes(cut_base_mesh, cut_base_mesh_boundary_verts,
                                            cut_fusion_mesh, cut_fusion_mesh_boundary_verts)
        connected_mesh = connect_output["connected_mesh"]
        face_mask_connect = connect_output["valid_face_mask"]
        connected_max_dist = connect_output["max_dist"]
        max_dist_in_connection = np.maximum(max_dist_in_connection, connected_max_dist)

        if mesh_output_dir is not None:
            cut_fusion_mesh.export(merge_output_dir + f"cut_fusion_mesh_{cc_label}_debug.obj")
            cut_base_mesh.export(merge_output_dir + f"cut_base_mesh_{cc_label}_debug.obj")

        if force_watertight and (not connected_mesh.is_watertight):
            print(f"not water tight for {cc_label}")
            if mesh_output_dir is not None:
                connected_mesh.export(merge_output_dir + f"connected_mesh_{cc_label}_notWT.obj")
            continue  # skip updating base_mesh

        if force_short_edge and connected_max_dist > 6 * base_edge_average_len:
            print(f"too long distance for connecting {cc_label}: {connected_max_dist} vs. {4 * base_edge_average_len}.")
            if mesh_output_dir is not None:
                connected_mesh.export(merge_output_dir + f"connected_mesh_{cc_label}_bad.obj")
            # input()
            continue  # skip updating base_mesh

        connected_mesh.fill_holes()
        if mesh_output_dir is not None:
            connected_mesh.export(merge_output_dir + f"connected_mesh_{cc_label}.obj")
        
        face_mask_this_cc = np.ones((base_mesh.faces.shape[0]), dtype=bool)
        face_mask_this_cc[~cut_base_mesh_face_mask] = False  # deleted by cutting
        face_mask_this_cc[cut_base_mesh_face_mask] = face_mask_connect[:cur_base_mesh_face_num]  # deleted by degeneration

        base_mesh = connected_mesh.copy()
        # assert base_mesh.faces.shape[0] == face_mask_this_cc.sum() + face_mask_connect[cur_base_mesh_face_num:].sum()

        track_face_num = track_face_mask.sum()   # the first x faces are from the very beginning mesh
        track_face_mask[track_face_mask] = face_mask_this_cc[:track_face_num]

    cc_update_num = np.sum(cc_success_flag)

    if mesh_output_dir is not None:
        log_path = mesh_output_dir + f"merge/log_pad{int(aabb_pad * 1000):d}mm.json"
    else:
        assert (log_out_dir is not None)
        log_path = log_out_dir + "updated_mesh_log.json"

    log_dict.update({"cc_update_num": int(cc_update_num), "max_dist_in_connection": float(max_dist_in_connection)})
    log_json = json.dumps(log_dict, sort_keys=True, indent=4, separators=(',', ': '))
    fout = open(log_path, 'w')
    fout.write(log_json)
    fout.close()

    if cc_update_num == 0:
        return {"cc_update_num": 0}

    new_ref_area = base_mesh.area_faces.copy()
    track_face_num = track_face_mask.sum()
    new_ref_area[:track_face_num] = base_mesh_ori.area_faces[track_face_mask]
    new_face_area_average = np.average(new_ref_area[track_face_num:])
    new_ref_area[track_face_num:] = new_face_area_average

    return {"updated_mesh": base_mesh,
            "cc_update_num": cc_update_num,
            "track_face_mask": track_face_mask,
            "new_ref_area": new_ref_area,
            "max_dist_in_connection": max_dist_in_connection}


@torch.no_grad()
def detect_topo_err(refined_sugar: SuGaR, nerfmodel: GaussianSplattingWrapper, work_dir, cmr, ite,
                    use_depth_loss=True, depth_scalar=1, use_color_loss=True, color_scalar=1,
                    use_densifier_grad=False, grad_scalar=1, use_opacity_loss=False,
                    save_inter=False, save_render=False, save_mesh=True,
                    mesh_prop=False, detect_floor=True, min_observe=4, voxel_size=0.01):
    # if cmr is None:
    #     cmr = np.load(data_root + "rgb_cameras.npz")
    intr = cmr["intrinsics"]
    extr = cmr["extrinsics"]
    shape = cmr["shape"]
    cmr_num = shape.shape[0]

    vert, face, face_color = refined_sugar.get_color_mesh()
    mesh = trimesh.Trimesh(vertices=vert, faces=face, face_colors=face_color, process=False)

    vis_dir = work_dir + f"detect/visual_{ite:06d}/"
    if save_inter:
        os.makedirs(vis_dir, exist_ok=True)

    render_dir = work_dir + f"detect/render_{ite:06d}/"
    if save_render:
        os.makedirs(render_dir, exist_ok=True)

    mesh_dir = work_dir + f"detect/mesh_{ite:06d}/"
    if save_mesh:
        os.makedirs(mesh_dir, exist_ok=True)

    vert_num = mesh.vertices.shape[0]
    vert_depth_loss_total = np.zeros((cmr_num, vert_num))
    vert_color_loss_total = np.zeros((cmr_num, vert_num))
    vert_visual_total = np.zeros_like(vert_depth_loss_total, dtype=bool)

    for cmr_idx in tqdm(range(0, cmr_num)):

        # ---------- RENDER ----------
        # RGB
        rgb = refined_sugar.render_image_gaussian_rasterizer(
            camera_indices=cmr_idx,
            bg_color=[0.0, 1.0, 0.0],
            sh_deg=nerfmodel.gaussians.active_sh_degree,
            compute_color_in_rasterizer=True,
            compute_covariance_in_rasterizer=True,
            return_2d_radii=False,
            use_same_scale_in_all_directions=False,
        ).clamp(min=0., max=1.).contiguous()
        render_color = rgb.cpu().numpy()[..., ::-1]
        if save_render:
            cv2.imwrite(render_dir + f"/render_{cmr_idx:06d}.jpg", render_color * 255.0)

        # depth
        fov_camera = nerfmodel.training_cameras.p3d_cameras[cmr_idx]
        point_depth = fov_camera.get_world_to_view_transform().transform_points(refined_sugar.points)[..., 2:].expand(-1, 3)

        render_depth = refined_sugar.render_image_gaussian_rasterizer(
            camera_indices=cmr_idx,
            bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=refined_sugar.device),
            sh_deg=0,
            compute_color_in_rasterizer=False,
            compute_covariance_in_rasterizer=True,
            use_same_scale_in_all_directions=False,
            point_colors=point_depth,
        ).contiguous()
        render_depth = render_depth.cpu().numpy()[..., 0]
        if save_render:
            render_depth_vis = np.uint8(render_depth / max_depth * 255.0)
            render_depth_vis = cv2.applyColorMap(render_depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(render_dir + f"/depth_{cmr_idx:06d}.jpg", render_depth_vis)

        surface_depth = refined_sugar.render_image_gaussian_rasterizer(
            camera_indices=cmr_idx,
            bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=refined_sugar.device),
            sh_deg=0,
            compute_color_in_rasterizer=False,
            compute_covariance_in_rasterizer=True,
            use_solid_surface=True,  # DIFFERENCE
            use_same_scale_in_all_directions=False,
            point_colors=point_depth,
        ).contiguous()
        surface_depth = surface_depth.cpu().numpy()[..., 0]

        if save_render:
            surface_depth_vis = np.uint8(surface_depth / max_depth * 255.0)
            surface_depth_vis = cv2.applyColorMap(surface_depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(render_dir + f"/depth_sur_{cmr_idx:06d}.jpg", surface_depth_vis)

            surface_depth_mm = np.uint16(surface_depth * 1000)
            cv2.imwrite(render_dir + f"/depth_sur_mm_{cmr_idx:06d}.png", surface_depth_mm)

        # ---------- COMPUTE LOSS FOR VERT ----------
        # Load GT depth map
        depth_gt = nerfmodel.get_gt_depth(camera_indices=cmr_idx)[..., 0].cpu().numpy()
        edge_depth_gt = get_depth_edge(depth_gt, 3)

        depth_diff = np.abs(np.minimum(depth_gt, max_depth) - render_depth)
        if save_inter:
            depth_diff_vis = np.minimum(depth_diff * 10, 1)
            cv2.imwrite(vis_dir + f"depth_diff_{cmr_idx:06d}.jpg", depth_diff_vis * 255.0)

        # Project vertices to depth image
        pix_v, local_points = project(mesh.vertices, intr[cmr_idx], extr[cmr_idx], shape[cmr_idx],
                                      return_local_points=True)
        pix_depth, valid_mask = query_at_image(surface_depth, pix_v, return_valid=True)
        visual_mask = valid_mask & (np.abs(local_points[..., 2] - pix_depth) < 0.005)

        edge_vis = np.minimum(edge_depth_gt / edge_depth_gt.max() * 1000, 1)
        edge_weight_cur = query_at_image(edge_vis, pix_v)
        visual_mask = visual_mask & (edge_weight_cur < 0.1)
        vert_visual_total[cmr_idx] = visual_mask
        if save_inter:
            cv2.imwrite(vis_dir + f"edge_vis_{cmr_idx:04d}.jpg", edge_vis * 255.0)

        if use_depth_loss:
            depth_loss_map = np.minimum(depth_diff * (1 - edge_vis) * 10, 2)  # --- TODO: IMPROVE ---
            if save_inter:
                cv2.imwrite(vis_dir + f"depth_diff_vis_{cmr_idx:04d}.jpg", np.minimum(depth_loss_map * 255.0, 255))
            vert_depth_loss = query_at_image(depth_loss_map, pix_v, return_valid=False)
            vert_depth_loss_total[cmr_idx][visual_mask] = vert_depth_loss[visual_mask]

        if use_color_loss:
            color_gt = nerfmodel.get_gt_image(camera_indices=cmr_idx).cpu().numpy()[..., ::-1]
            color_loss_map = np.abs(color_gt - render_color) * 255
            color_loss_map = np.average(color_loss_map, axis=-1)
            color_loss_map[depth_gt > max_depth] = 0

            color_loss_map = color_loss_map * (1 - edge_vis) * 3
            if save_inter:
                cv2.imwrite(vis_dir + f"color_diff_vis_{cmr_idx:04d}.jpg", np.minimum(color_loss_map, 255))
            vert_color_loss = query_at_image(color_loss_map, pix_v, return_valid=False)
            vert_color_loss_total[cmr_idx][visual_mask] = vert_color_loss[visual_mask]

    vert_visual_cnt = np.sum(np.int32(vert_visual_total), axis=0)
    vert_depth_loss_average = np.zeros(vert_num)
    vert_color_loss_average = np.zeros(vert_num)

    for v_idx in range(vert_num):
        if vert_visual_cnt[v_idx] >= min_observe:
            v_visual_cmr = vert_visual_total[:, v_idx]

            if use_depth_loss:
                vert_depth_loss_list = vert_depth_loss_total[v_visual_cmr, v_idx]
                vert_depth_loss_average[v_idx] = np.average(vert_depth_loss_list, axis=0)

            if use_color_loss:
                vert_color_loss_list = vert_color_loss_total[v_visual_cmr, v_idx]
                vert_color_loss_average[v_idx] = np.average(vert_color_loss_list, axis=0)

    vert_loss_sum = np.zeros((vert_num, 3))

    if use_depth_loss:
        vert_depth_loss_vis = vert_depth_loss_average[..., None].repeat(3, axis=1)
        if save_mesh:
            depth_loss_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                                              vertex_colors=np.minimum(vert_depth_loss_vis * 255, 255).astype(int), process=False)
            depth_loss_mesh.export(mesh_dir + f"depth_loss.obj")
        vert_depth_loss_vis = vert_depth_loss_vis * depth_scalar
        vert_loss_sum += vert_depth_loss_vis

    if use_color_loss:
        vert_color_loss_vis = vert_color_loss_average[..., None].repeat(3, axis=1)
        if save_mesh:
            pc = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vert_color_loss_vis.astype(int), process=False)
            pc.export(mesh_dir + f"color_loss.obj")
        vert_color_loss_vis = vert_color_loss_vis * color_scalar
        vert_loss_sum += vert_color_loss_vis

    if use_opacity_loss:
        face_opacity = refined_sugar.strengths.reshape(face.shape[0], refined_sugar.n_gaussians_per_surface_triangle)
        face_opacity = face_opacity.mean(axis=-1, keepdim=True).cpu().numpy()
        # face_opacity = face_opacity.min(axis=-1, keepdim=True).cpu().numpy()
        face_opacity_loss = np.maximum(0.8 - face_opacity, 0) * 10
        face_opacity_loss = np.minimum(face_opacity_loss, 1)
        if save_mesh:
            opacity_loss_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                                                face_colors=face_opacity_loss.repeat(3, axis=1) * 255, process=False)
            opacity_loss_mesh.export(mesh_dir + f"opacity_loss.obj")

    if use_densifier_grad:
        if os.path.exists(work_dir + f"detect/densifier_grad_1000.obj"):
            mesh_grad_vis = trimesh.load_mesh(work_dir + f"detect/densifier_grad_1000.obj")
            vert_grad_vis = mesh_grad_vis.visual.vertex_colors[..., :3]
            vert_loss_sum += vert_grad_vis * grad_scalar
        else:
            print("Positional gradient file not exist, skip using positional gradient for topo change detection.")

    # Post-processing
    if detect_floor:
        # face_center = np.mean(mesh.vertices[mesh.faces], axis=1)
        vert_y = mesh.vertices[:, 1]
        floor_y = vert_y.min()
        floor_vert_mask = vert_y < floor_y + 0.02
        vert_loss_sum[floor_vert_mask] = 0
        vert_visual_cnt[floor_vert_mask] = min_observe + 1

    if save_mesh:
        detect_loss_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                                           vertex_colors=np.minimum(vert_loss_sum * 255, 255).astype(int), process=False)
        detect_loss_mesh.export(mesh_dir + f"detect_vc.obj")
        # face_depth_loss = detect_loss_mesh.visual.face_colors
        # detect_loss_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=face_depth_loss, process=False)
        # detect_loss_mesh.export(mesh_dir + f"depth_diff_fc.obj")

    if mesh_prop:
        mesh_vert_propagate(mesh.vertex_neighbors, vert_visual_cnt >= min_observe,
                            value=vert_loss_sum, max_ite=mesh_prop)
        if save_mesh:
            detect_loss_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                                               vertex_colors=np.minimum(vert_loss_sum * 255, 255).astype(int), process=False)
            detect_loss_mesh.export(mesh_dir + f"detect_prop_vc.obj")
            # face_depth_loss = detect_loss_mesh.visual.face_colors
            # detect_loss_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=face_depth_loss, process=False)
            # detect_loss_mesh.export(mesh_dir + f"depth_diff_prop_fc.obj")

    voxel_center, voxel_color = build_voxel_from_pc(mesh.vertices, vert_loss_sum, voxel_size)
    vert_loss_sum = interpolate_in_voxel(mesh.vertices, voxel_center, voxel_color, voxel_size, knn_K=8)
    detect_loss_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                                       vertex_colors=np.minimum(vert_loss_sum * 255, 255).astype(int), process=False)
    face_loss = detect_loss_mesh.visual.face_colors
    if save_mesh:
        detect_loss_mesh.export(mesh_dir + f"detect_voxel_vc.obj")
        detect_loss_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                                          face_colors=face_loss, process=False)
        detect_loss_mesh.export(mesh_dir + f"detect_voxel_fc.obj")

    return face_loss[..., 0] / 255



def forward_rendering_and_mesh_update(args):
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

    # --- Output parameters ---
    mesh_output_dir = args.mesh_output_dir
    os.makedirs(mesh_output_dir, exist_ok=True)
    mesh_save_path = os.path.join(mesh_output_dir, 'color_mesh.obj')
    sugar_mesh_path = args.mesh_path

    # Postprocessing
    postprocess_mesh = args.postprocess_mesh
    if postprocess_mesh:
        postprocess_density_threshold = args.postprocess_density_threshold
        postprocess_iterations = args.postprocess_iterations

    CONSOLE.print('==================================================')
    CONSOLE.print("Starting extracting texture from refined SuGaR model:")
    CONSOLE.print('Scene path:', source_path)
    CONSOLE.print('Iteration to load:', iteration_to_load)
    CONSOLE.print('Vanilla 3DGS checkpoint path:', gs_checkpoint_path)
    CONSOLE.print('Refined model path:', refined_model_path)
    CONSOLE.print('Coarse mesh path:', sugar_mesh_path)
    CONSOLE.print('Mesh output directory:', mesh_output_dir)
    CONSOLE.print('Mesh save path:', mesh_save_path)
    CONSOLE.print('Number of gaussians per surface triangle:', n_gaussians_per_surface_triangle)
    CONSOLE.print('Postprocess mesh:', postprocess_mesh)
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
        from_humanrf=args.from_humanrf,
        )
    CONSOLE.print("Vanilla 3DGS Loaded.")
    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')
    CONSOLE.print(len(nerfmodel.gaussians._xyz) / 1e6, "M gaussians detected.")

    # ZCW change
    nerfmodel.gaussians.active_sh_degree = 2  # SH: 2 or 3

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

    # Mesh update
    if args.enable_mesh_update and args.enable_unbind:
        # vert, face, face_color = refined_sugar.get_color_mesh()
        fusion_mesh = extract_mesh_fusion(refined_sugar, nerfmodel,
                                          # save_dir=(mesh_output_dir + "extract/"),
                                          # simplify_face_num=np.array(o3d_mesh.triangles).shape[0],
                                          smooth=False)
        o3d.io.write_triangle_mesh(mesh_output_dir + "extract_sdf_alpha.obj", fusion_mesh)
        fusion_mesh = trimesh.load_mesh(mesh_output_dir + "extract_sdf_alpha.obj")
        # base_mesh = trimesh.load_mesh(mesh_output_dir + "color_mesh.obj")
        detected_mesh = trimesh.load_mesh(mesh_output_dir + "detect/mesh_001000/detect_voxel_fc.obj")

        # Try different aabb pads and select the best
        aabb_pad_arr = np.array([0.01, 0.015, 0.02, 0.025, 0.03])
        max_dist_arr = np.ones_like(aabb_pad_arr) * 100
        nothing_to_update = False
        for i in range(aabb_pad_arr.shape[0]):
            update_output = update_mesh_topo(refined_sugar, fusion_mesh, aabb_pad=aabb_pad_arr[i],
                                             mesh_output_dir=mesh_output_dir, highlight_boundary=True,
                                             detected_mesh=detected_mesh, force_watertight=args.force_watertight)
            if update_output["cc_update_num"] == -1:
                nothing_to_update = True
                break
            elif update_output["cc_update_num"] > 0:
                max_dist_arr[i] = update_output["max_dist_in_connection"]

        if not nothing_to_update:
            best_aabb_pad = aabb_pad_arr[np.argmin(max_dist_arr)]
            print(f"best_aabb_pad: {best_aabb_pad}")
            update_output = update_mesh_topo(refined_sugar, fusion_mesh, aabb_pad=best_aabb_pad,
                                             log_out_dir=mesh_output_dir,
                                             detected_mesh=detected_mesh, force_watertight=args.force_watertight)

            if update_output["cc_update_num"] > 0:
                updated_mesh = update_output["updated_mesh"]
                updated_mesh.export(mesh_output_dir + "updated_mesh.obj")

                np.savez_compressed(mesh_output_dir + "face_corr.npz",
                                    track_face_mask=update_output["track_face_mask"],
                                    ref_area=update_output["new_ref_area"])

    # ZCW rendering
    if args.render_results is not None:
        cmr_num = len(nerfmodel.training_cameras)

        if args.render_results[0] == 'w':
            render_dir = mesh_output_dir + f"render_w/"
            bg_color = [1.0, 1.0, 1.0]
        elif args.render_results[0] == 'b':
            render_dir = mesh_output_dir + f"render_b/"
            bg_color = [0.0, 0.0, 0.0]
        else:
            render_dir = mesh_output_dir + f"render/"
            bg_color = [0.0, 1.0, 0.0]
        os.makedirs(render_dir, exist_ok=True)

        if len(args.render_results) > 1:
            save_depth = True
        else:
            save_depth = False

        print("Start rendering")
        with (torch.no_grad()):
            for cmr_idx in tqdm(range(cmr_num)):
            # for cmr_idx in tqdm(range(15, 20)):
                # RGB
                rgb = refined_sugar.render_image_gaussian_rasterizer(
                    camera_indices=cmr_idx,
                    bg_color=bg_color,
                    sh_deg=nerfmodel.gaussians.active_sh_degree,
                    compute_color_in_rasterizer=True,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=False,
                    use_same_scale_in_all_directions=False,
                ).clamp(min=0., max=1.).contiguous()

                render_img = rgb.cpu().numpy()[..., ::-1]
                cv2.imwrite(render_dir + f"/render_{cmr_idx:06d}.jpg", render_img * 255.0)

                if not save_depth:
                    continue

                # DEPTH
                fov_camera = nerfmodel.training_cameras.p3d_cameras[cmr_idx]
                point_depth = fov_camera.get_world_to_view_transform().transform_points(refined_sugar.points)[..., 2:].expand(-1, 3)
                render_depth = refined_sugar.render_image_gaussian_rasterizer(
                    camera_indices=cmr_idx,
                    bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=refined_sugar.device),
                    sh_deg=0,
                    compute_color_in_rasterizer=False,
                    compute_covariance_in_rasterizer=True,
                    use_same_scale_in_all_directions=False,
                    point_colors=point_depth,
                ).contiguous()
                render_depth = render_depth.cpu().numpy()[..., 0]
                render_depth_vis = np.uint8(render_depth / max_depth * 255.0)
                render_depth_vis = cv2.applyColorMap(render_depth_vis, cv2.COLORMAP_JET)
                cv2.imwrite(render_dir + f"/depth_{cmr_idx:06d}.jpg", render_depth_vis)
                # if 1:
                #     render_depth_mm = np.uint16(render_depth * 1000)
                #     cv2.imwrite(render_dir + f"/depth_mm_{cmr_idx:06d}.png", render_depth_mm)

                # SURFACE DEPTH
                surface_depth = refined_sugar.render_image_gaussian_rasterizer(
                    camera_indices=cmr_idx,
                    bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=refined_sugar.device),
                    sh_deg=0,
                    compute_color_in_rasterizer=False,
                    compute_covariance_in_rasterizer=True,
                    use_solid_surface=True,  # DIFFERENCE
                    use_same_scale_in_all_directions=False,
                    point_colors=point_depth,
                ).contiguous()
                surface_depth = surface_depth.cpu().numpy()[..., 0]
                surface_depth_vis = np.uint8(surface_depth / max_depth * 255.0)
                surface_depth_vis = cv2.applyColorMap(surface_depth_vis, cv2.COLORMAP_JET)
                cv2.imwrite(render_dir + f"/depth_sur_{cmr_idx:06d}.jpg", surface_depth_vis)
                # if 1:
                #     surface_depth_mm = np.uint16(surface_depth * 1000)
                #     cv2.imwrite(render_dir + f"/depth_sur_mm_{cmr_idx:06d}.png", surface_depth_mm)

                if args.save_diff:
                    depth_gt = np.minimum(nerfmodel.get_gt_depth(camera_indices=cmr_idx).cpu().numpy(), max_depth).squeeze()
                    # # depth_gt_vis = np.minimum(depth_gt / max_depth, 1)
                    # depth_gt_vis = np.uint8(depth_gt / max_depth * 255.0)
                    # depth_gt_vis = cv2.applyColorMap(depth_gt_vis, cv2.COLORMAP_JET)
                    # cv2.imwrite(render_dir + f"/depth_gt_{cmr_idx:06d}.jpg", depth_gt_vis)

                    depth_diff = np.abs(depth_gt - render_depth)
                    depth_diff = np.minimum(depth_diff / max_depth * 100, 1)
                    cv2.imwrite(render_dir + f"/depth_diff_{cmr_idx:06d}.jpg", np.uint8(depth_diff * 255.0))
                    # diff_vis = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
                    # cv2.imwrite(render_dir + f"/depth_diff_{cmr_idx:06d}_jets.jpg", diff_vis)

    if postprocess_mesh:
        CONSOLE.print("Postprocessing mesh by removing border triangles with low-opacity gaussians...")
        with torch.no_grad():
            new_verts = refined_sugar.surface_mesh.verts_list()[0].detach().clone()
            new_faces = refined_sugar.surface_mesh.faces_list()[0].detach().clone()
            new_normals = refined_sugar.surface_mesh.faces_normals_list()[0].detach().clone()

            # For each face, get the 3 edges
            edges0 = new_faces[..., None, (0,1)].sort(dim=-1)[0]
            edges1 = new_faces[..., None, (1,2)].sort(dim=-1)[0]
            edges2 = new_faces[..., None, (2,0)].sort(dim=-1)[0]
            all_edges = torch.cat([edges0, edges1, edges2], dim=-2)

            # We start by identifying the inside faces and border faces
            face_mask = refined_sugar.strengths[..., 0] > -1.
            for i in range(postprocess_iterations):
                CONSOLE.print("\nStarting postprocessing iteration", i)
                # We look for edges that appear in the list at least twice (their NN is themselves)
                edges_neighbors = knn_points(all_edges[face_mask].view(1, -1, 2).float(), all_edges[face_mask].view(1, -1, 2).float(), K=2)
                # If all edges of a face appear in the list at least twice, then the face is inside the mesh
                is_inside = (edges_neighbors.dists[0][..., 1].view(-1, 3) < 0.01).all(-1)
                # We update the mask by removing border faces
                face_mask[face_mask.clone()] = is_inside

            # We then add back border faces with high-density
            face_centers = new_verts[new_faces].mean(-2)
            face_densities = refined_sugar.compute_density(face_centers[~face_mask])
            face_mask[~face_mask.clone()] = face_densities > postprocess_density_threshold

            # And we create the new mesh and SuGaR model
            new_faces = new_faces[face_mask]
            new_normals = new_normals[face_mask]

            new_scales = refined_sugar._scales.reshape(len(face_mask), -1, 2)[face_mask].view(-1, 2)
            new_quaternions = refined_sugar._quaternions.reshape(len(face_mask), -1, 2)[face_mask].view(-1, 2)
            new_densities = refined_sugar.all_densities.reshape(len(face_mask), -1, 1)[face_mask].view(-1, 1)
            new_sh_coordinates_dc = refined_sugar._sh_coordinates_dc.reshape(len(face_mask), -1, 1, 3)[face_mask].view(-1, 1, 3)
            new_sh_coordinates_rest = refined_sugar._sh_coordinates_rest.reshape(len(face_mask), -1, 15, 3)[face_mask].view(-1, 15, 3)

            new_o3d_mesh = o3d.geometry.TriangleMesh()
            new_o3d_mesh.vertices = o3d.utility.Vector3dVector(new_verts.cpu().numpy())
            new_o3d_mesh.triangles = o3d.utility.Vector3iVector(new_faces.cpu().numpy())
            new_o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(new_normals.cpu().numpy())
            new_o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(torch.ones_like(new_verts).cpu().numpy())

            refined_sugar = SuGaR(
                nerfmodel=nerfmodel,
                points=None,
                colors=None,
                initialize=False,
                sh_levels=nerfmodel.gaussians.active_sh_degree+1,
                keep_track_of_knn=False,
                knn_to_track=0,
                beta_mode='average',
                surface_mesh_to_bind=new_o3d_mesh,
                n_gaussians_per_surface_triangle=refined_sugar.n_gaussians_per_surface_triangle,
                )
            refined_sugar._scales[...] = new_scales
            refined_sugar._quaternions[...] = new_quaternions
            refined_sugar.all_densities[...] = new_densities
            refined_sugar._sh_coordinates_dc[...] = new_sh_coordinates_dc
            refined_sugar._sh_coordinates_rest[...] = new_sh_coordinates_rest
        CONSOLE.print("Mesh postprocessed.")

    # Compute texture
    if args.UV_texture:
        raise RuntimeError("SuGaR texture does not apply for GauSTAR")

    elif args.mesh_extraction:
        vert, face, face_color = refined_sugar.get_color_mesh()
        obj_mesh = trimesh.Trimesh(vertices=vert, faces=face, face_colors=face_color, process=False)
        obj_mesh.export(mesh_save_path)

        CONSOLE.print("Color mesh saved at:", mesh_save_path)

    return mesh_save_path
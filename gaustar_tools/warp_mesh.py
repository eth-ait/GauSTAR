import os

import numpy as np
import argparse
import trimesh
import cv2
from tqdm import tqdm
import open3d as o3d
import torch
from pytorch3d.ops import knn_points

import json

class warp_config:
    min_observe = 4
    depth_edge_ker_size = 7
    knn_K = 8
    cmr_view_max_cos = -0.5
    max_move_dist = 0.2
    voxel_size = 0.04
    bi_direct_pix_threshold = 4
    bi_direct_depth_threshold = 0.004
    edge_scalar = 10000
    edge_threshold = 0.1
    post_processing = 'mesh'

    def save_cfg(self, dir):
        cfg_dict = {"min_observe": self.min_observe,
                    "depth_edge_ker_size": self.depth_edge_ker_size,
                    "knn_K": self.knn_K,
                    "cmr_view_max_cos": self.cmr_view_max_cos,
                    "max_move_dist": self.max_move_dist,
                    "voxel_size": self.voxel_size,
                    "bi_direct_pix_threshold": self.bi_direct_pix_threshold,
                    "bi_direct_depth_threshold": self.bi_direct_depth_threshold,
                    "edge_scalar": self.edge_scalar,
                    "edge_threshold": self.edge_threshold,
                    "post_processing": self.post_processing,
                    }

        cfg_json = json.dumps(cfg_dict, sort_keys=True, indent=4, separators=(',', ': '))

        fout = open(os.path.join(dir, 'config.json'), 'w')
        fout.write(cfg_json)
        fout.close()


def points_to_local_points(points: np.ndarray, extr):
    rot = extr[:3, :3]
    trans = extr[:3, 3, None]
    local_points = (np.matmul(rot, points.T) + trans).T
    # pc = trimesh.Trimesh(vertices=local_points)
    # pc.export("out/local_points.ply")
    return local_points


def project(points: np.ndarray, intr, extr, shape, return_local_points=False):
    """Projects a 3D point (x,y,z) to a pixel position (x,y)."""
    batch_shape = points.shape[:-1]
    points = points.reshape((-1, 3))

    local_points = points_to_local_points(points, extr)
    x = local_points[..., 0] / local_points[..., 2]
    y = local_points[..., 1] / local_points[..., 2]

    pixel_c = intr[0, 0] * x + shape[1] * 0.5  # col
    pixel_r = intr[1, 1] * y + shape[0] * 0.5  # row
    pixels = np.stack([pixel_r, pixel_c], axis=-1)

    if return_local_points:
        # depth = np.linalg.norm(local_points, axis=-1, keepdims=True)
        return pixels.reshape((*batch_shape, 2)), local_points
    else:
        return pixels.reshape((*batch_shape, 2))


def pixel_to_local_rays(pixels: np.ndarray, intr, shape):
    """Returns the local ray directions for the provided pixels."""
    x = (pixels[..., 1] - shape[1] * 0.5) / intr[0, 0]
    y = (pixels[..., 0] - shape[0] * 0.5) / intr[1, 1]

    dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    return dirs  # / np.linalg.norm(dirs, axis=-1, keepdims=True)


def pixels_to_points(pixels: np.ndarray, depth: np.ndarray, intr, extr, shape):
    rays_through_pixels = pixel_to_local_rays(pixels, intr, shape)
    local_points = rays_through_pixels * depth[..., None]
    # return local_points
    rot = extr[:3, :3]
    trans = extr[:3, 3]
    points = np.matmul(rot.T, (local_points - trans).T)
    return points.T


def pad_and_resize_flow(flow, pad, shape):
    if pad is not None:
        pad = (np.append(pad, [0, 0])).reshape((-1, 2))
        flow = np.pad(flow, pad_width=np.int32(pad), mode='constant', constant_values=0)
    scalar = shape[0] / flow.shape[0]
    flow *= scalar
    flow = cv2.resize(flow, shape[::-1], interpolation=cv2.INTER_NEAREST)
    return flow


def query_at_image(image, pix, return_valid=False):
    assert pix.shape[1] == 2
    pix = np.int32(pix + 0.5)
    shape = np.int32(image.shape[:2]) - 1
    pix_clip = np.clip(pix, a_min=0, a_max=shape)

    if return_valid:
        valid_mask = (pix == pix_clip)
        valid_mask = valid_mask[..., 0] & valid_mask[..., 1]
        return image[pix_clip[:, 0], pix_clip[:, 1]], valid_mask
    else:
        return image[pix_clip[:, 0], pix_clip[:, 1]]


def get_depth_edge(depth, ker_size=9, max_depth=None):
    if max_depth is None:
        depth_valid = depth[depth < 10]
        max_depth = depth_valid.max() * 1.1
    depth = np.minimum(depth, max_depth)
    mean = cv2.blur(depth, (ker_size, ker_size))
    mean_seq = mean ** 2
    seq = depth ** 2
    seq_mean = cv2.blur(seq, (ker_size, ker_size))
    var = np.maximum(seq_mean - mean_seq, 0)
    return var


def mesh_vert_propagate(vertex_neighbors, valid_mask, value, max_ite=20, debug_print=True):
    for ite in range(max_ite):
        prop_idx = np.array(np.where(valid_mask == False))[0]
        if debug_print:
            print(f"Iteration {ite} has {prop_idx.size} vertices to be propagated")

        cnt = 0
        new_valid_mask = valid_mask.copy()
        for v_idx in prop_idx:
            neighbor_idx = np.array(vertex_neighbors[v_idx])
            neighbor_valid_mask = valid_mask[neighbor_idx]
            if np.any(neighbor_valid_mask):
                valid_heighbor_idx = neighbor_idx[neighbor_valid_mask]
                neighbor_value = value[valid_heighbor_idx]
                value[v_idx] = np.average(neighbor_value, axis=0)
                new_valid_mask[v_idx] = True
                cnt += 1
        valid_mask = new_valid_mask

        if cnt == 0:
            if debug_print:
                print("Done. Nothing to propagate.")
            break
    return value

def mesh_color_smoothing(vertex_neighbors, vertex_color, ite_num=10, valid_mask=None):
    vert_num = len(vertex_neighbors)
    for ite in range(ite_num):
        new_vertex_color = vertex_color.copy()
        for v_idx in range(vert_num):
            neighbor_idx = np.array(vertex_neighbors[v_idx])
            # if valid_mask is not None:
            #     neighbor_valid_mask = valid_mask[neighbor_idx]
            #     if np.any(neighbor_valid_mask):
            #         valid_heighbor_idx = neighbor_idx[neighbor_valid_mask]
            neighbor_color = vertex_color[neighbor_idx]
            new_vertex_color[v_idx] = np.average(neighbor_color, axis=0)
        vertex_color = new_vertex_color
    return vertex_color


def remove_outlier(data, threshold=2, max_std=None):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    if max_std is not None:
        std = np.minimum(std, max_std)
    z = (data - mean) / std
    idx = (np.sum(z < threshold, axis=1) == 3)
    return data[idx]


def build_voxel_from_pc(pc_points, pc_color, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_points)
    pcd.colors = o3d.utility.Vector3dVector(pc_color)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    voxels = voxel_grid.get_voxels()

    voxel_num = len(voxels)
    voxel_center = np.zeros((voxel_num, 3))
    for vx_idx in range(voxel_num):
        voxel_center[vx_idx] = voxel_grid.get_voxel_center_coordinate(voxels[vx_idx].grid_index)
    voxel_color = np.array([vx.color for vx in voxels])
    return voxel_center, voxel_color


def interpolate_in_voxel(points, voxel_center, voxel_color, voxel_size, knn_K):
    knn = knn_points(torch.from_numpy(points[None]).float(), torch.from_numpy(voxel_center[None]).float(), K=knn_K)
    knn_idx = np.int32(knn.idx[0])
    knn_dist = knn.dists[0].numpy()

    points_color = np.zeros_like(points)
    for vert_idx in range(points.shape[0]):
        # vx_idx = int(knn_idx[vert_idx])
        # vert_move_average[vert_idx] = voxels[vx_idx].color
        vx_knn_color = voxel_color[knn_idx[vert_idx]]
        vx_knn_dist = knn_dist[vert_idx]
        vx_knn_weight = np.exp(-vx_knn_dist / (voxel_size ** 2)) + 1e-8  # or 1 / vx_knn_dist
        points_color[vert_idx] = np.average(vx_knn_color, axis=0, weights=vx_knn_weight)

    return points_color


def warp_mesh_using_flow(mesh_path, data_root, work_root, f_idx, interval=1,
                         cmr=None, save_inter=False, from_humanrf=False):
    if cmr is None:
        cmr = np.load(data_root + "rgb_cameras.npz")
    intr = cmr["intrinsics"]
    extr = cmr["extrinsics"]
    shape = cmr["shape"]
    cmr_num = shape.shape[0]

    cfg = warp_config()
    out_dir = work_root + f"{(f_idx+interval):04d}/coarse_mesh/"
    os.makedirs(out_dir, exist_ok=True)
    cfg.save_cfg(out_dir)

    mesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=True)
    if not hasattr(mesh.visual, 'vertex_colors'):
        mesh.visual = mesh.visual.to_color()  # UV texture to vertex colors

    vert_num = mesh.vertices.shape[0]
    vert_move_total = np.zeros((cmr_num, vert_num, 3))
    vert_dist_total = np.zeros((cmr_num, vert_num))
    vert_visual_total = np.zeros_like(vert_dist_total, dtype=bool)

    if save_inter:
        os.makedirs(out_dir + "visual/", exist_ok=True)

    if interval == 1:
        flow_dir = 'flow_bi'
    elif interval == 2:
        flow_dir = 'flow_bi_2f'
    elif interval == 4:
        flow_dir = 'flow_bi_4f'
    elif interval == 6:
        flow_dir = 'flow_bi_6f'
    else:
        raise RuntimeError("Interval Error!")

    if from_humanrf:
        label = "_humanrf"
    else:
        label = ""

    print("Scene flow warping for the next frame")
    for cmr_idx in tqdm(range(0, cmr_num)):
        # if cmr_idx in [5, 15, 25, 35, 45]:  # TEST set
        #     continue

        # Load optical flow
        if os.path.exists(data_root + f"{f_idx:04d}/{flow_dir}/pad.txt"):
            flow_pad = np.loadtxt(data_root + f"{f_idx:04d}/{flow_dir}/pad.txt")
        else:
            flow_pad = None

        if os.path.exists(data_root + f"{f_idx:04d}/{flow_dir}/{cmr_idx:04d}_f.npz"):
            flow = np.load(data_root + f"{f_idx:04d}/{flow_dir}/{cmr_idx:04d}_f.npz")['flow'][..., ::-1]
            flow_back = np.load(data_root + f"{f_idx:04d}/{flow_dir}/{cmr_idx:04d}_b.npz")['flow'][..., ::-1]
        else:
            raise RuntimeError("Flow not found!")
        flow = pad_and_resize_flow(flow, flow_pad, shape[cmr_idx])
        flow_back = pad_and_resize_flow(flow_back, flow_pad, shape[cmr_idx])

        # Load depth
        if os.path.exists(data_root + f"{f_idx:04d}/depth{label}/img_{cmr_idx:04d}_depth.npz"):
            depth_cur = np.load(data_root + f"{f_idx:04d}/depth{label}/img_{cmr_idx:04d}_depth.npz")['depth']
            depth_next = np.load(data_root + f"{(f_idx + interval):04d}/depth{label}/img_{cmr_idx:04d}_depth.npz")['depth']
        else:
            raise RuntimeError("Depth not found!")
        edge_cur = get_depth_edge(depth_cur, cfg.depth_edge_ker_size)
        edge_next = get_depth_edge(depth_next, cfg.depth_edge_ker_size)

        # Project vertices to depth image
        pix_cur, local_points = project(mesh.vertices, intr[cmr_idx], extr[cmr_idx], shape[cmr_idx],
                                        return_local_points=True)
        pix_depth_cur, valid_mask = query_at_image(depth_cur, pix_cur, return_valid=True)

        cmr_view_mesh = trimesh.Trimesh(vertices=local_points, faces=mesh.faces, process=False)
        # cmr_view_mesh.export(f"out/all/cmr_view_{cmr_idx:04d}.obj")
        cmr_view_normal_z = cmr_view_mesh.vertex_normals[..., 2]
        depth_diff_cur = np.abs(local_points[..., 2] - pix_depth_cur)
        visual_mask = valid_mask & (depth_diff_cur < 0.005) & (cmr_view_normal_z < cfg.cmr_view_max_cos)

        # Weight
        edge_vis = np.minimum(edge_cur / edge_cur.max() * cfg.edge_scalar, 1)
        edge_weight_cur = query_at_image(edge_vis, pix_cur)
        visual_mask = visual_mask & (edge_weight_cur < cfg.edge_threshold)

        # Move pixel according to optical flow
        pix_next = pix_cur + query_at_image(flow, pix_cur)

        # Compute confident base on bi-directional consistence
        pix_cur_back = pix_next + query_at_image(flow_back, pix_next)
        pix_depth_cur_back = query_at_image(depth_cur, pix_cur_back)
        visual_mask = visual_mask & (np.abs(pix_depth_cur_back - pix_depth_cur) < cfg.bi_direct_depth_threshold)
        pix_diff = np.linalg.norm((pix_cur_back-pix_cur), axis=-1)  # np.abs(pix_cur_back-pix_cur)
        visual_mask = visual_mask & (pix_diff < cfg.bi_direct_pix_threshold)

        # Weight in Next frame
        edge_vis_next = np.minimum(edge_next / edge_next.max() * cfg.edge_scalar, 1)
        edge_weight_next = query_at_image(edge_vis_next, pix_next)
        visual_mask = visual_mask & (edge_weight_next < cfg.edge_threshold)

        # Get depth in next frame
        pix_depth_next, valid_mask = query_at_image(depth_next, pix_next, return_valid=True)
        visual_mask = visual_mask & valid_mask & (pix_depth_next < 10)
        # visual_mask = visual_mask & ((pix_depth_next - pix_depth_cur) < 0.02)

        # local_visual_points = pixels_to_points(pix_next[visual_mask], pix_depth_next[visual_mask], intr[cmr_idx], extr[cmr_idx], shape[cmr_idx])

        # Compute 3d movement(flow)
        moved_points = pixels_to_points(pix_next, pix_depth_next, intr[cmr_idx], extr[cmr_idx], shape[cmr_idx])
        vert_move = moved_points - np.asarray(mesh.vertices)
        vert_move_dist = np.linalg.norm(vert_move, axis=-1)
        visual_mask = visual_mask & (vert_move_dist < cfg.max_move_dist)

        vert_move[~visual_mask] *= 0
        if save_inter:
            cv2.imwrite(out_dir + f"visual/edge_vis_{cmr_idx:04d}.jpg", edge_vis * 255.0)
            pc = trimesh.Trimesh(vertices=mesh.vertices + vert_move, faces=mesh.faces, process=False)
            pc.export(out_dir + f"visual/warp_visual_Cmr{cmr_idx:04d}.ply")

        # vert_move_average[visual_mask] += vert_move[visual_mask]
        # vert_visual_total[visual_mask] += 1
        vert_move_total[cmr_idx][visual_mask] = vert_move[visual_mask]
        vert_dist_total[cmr_idx][visual_mask] = vert_move_dist[visual_mask]
        vert_visual_total[cmr_idx] = visual_mask

    # np.save("out/vert_move_total.npy", vert_move_total)
    # vert_move_total = np.load("out/vert_move_total.npy")
    # np.save("out/vert_visual_total.npy", vert_visual_total)
    # vert_visual_total = np.load("out/vert_visual_total.npy")

    min_observe = cfg.min_observe

    vert_visual_cnt = np.sum(np.int32(vert_visual_total), axis=0)
    vert_move_average = np.zeros_like(mesh.vertices)
    for v_idx in range(vert_num):
        if vert_visual_cnt[v_idx] >= min_observe:
            v_visual_cmr = vert_visual_total[:, v_idx]
            # vert_move_average[v_idx] = np.average(vert_move_total[v_visual_cmr, v_idx], axis=0)
            vert_move_list = remove_outlier(vert_move_total[v_visual_cmr, v_idx])
            vert_visual_cnt[v_idx] = vert_move_list.shape[0]
            if vert_visual_cnt[v_idx] >= min_observe:
                vert_move_average[v_idx] = np.average(vert_move_list, axis=0)

    label = ""

    # vert_move_average = vert_move_average * cfg.move_scale

    pc = trimesh.Trimesh(vertices=(mesh.vertices + vert_move_average), faces=mesh.faces, process=False)
    pc.export(out_dir + f"warp_{f_idx:04d}{label}.obj")
    print("Saved at: ", out_dir + f"warp_{f_idx:04d}{label}.obj")

    if cfg.post_processing == 'voxel':
        visual_vert = vert_visual_cnt >= min_observe
        voxel_center, voxel_color = build_voxel_from_pc(mesh.vertices[visual_vert], vert_move_average[visual_vert], voxel_size=cfg.voxel_size)
        vert_move_average = interpolate_in_voxel(mesh.vertices, voxel_center, voxel_color, cfg.voxel_size, cfg.knn_K)

        pc = trimesh.Trimesh(vertices=(mesh.vertices + vert_move_average),
                             faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors, process=False)
        pc.export(out_dir + f"warp{label}_voxel.obj")
        print("Saved at: ", out_dir + f"warp{label}_voxel.obj")

    # vert_visual_total = np.maximum(vert_visual_total, 1)[..., None]
    # vert_move_average = vert_move_average / vert_visual_total

    # move_graph = trimesh.Trimesh(vertices=vert_move_average, faces=mesh.faces, process=False)

    if cfg.post_processing == 'mesh':
        vert_move_average = mesh_vert_propagate(pc.vertex_neighbors, vert_visual_cnt >= min_observe, value=vert_move_average)
        # vert_move_average = move_graph.vertices
        pc = trimesh.Trimesh(vertices=(mesh.vertices + vert_move_average),
                             faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors, process=False)
        pc.export(out_dir + f"warp{label}_mesh_prop.obj")
        # print("Saved at: ", out_dir + f"warp{label}_voxel.obj")
        # move_graph = trimesh.Trimesh(vertices=vert_move_average, faces=mesh.faces, process=False)

    # trimesh.smoothing.filter_laplacian(move_graph, iterations=10)
    # vert_move_average = move_graph.vertices
    vert_move_average = mesh_color_smoothing(pc.vertex_neighbors, vertex_color=vert_move_average, ite_num=5)
    pc = trimesh.Trimesh(vertices=(mesh.vertices + vert_move_average),
                         faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors, process=False)
    pc.export(out_dir + f"warp{label}_smooth.obj")
    # print("Saved at: ", out_dir + f"warp{label}_smooth.obj")

    # trimesh.smoothing.filter_laplacian(pc)
    # pc.export(work_root + f"{(f_idx+1):04d}/coarse_mesh/warp_{f_idx:04d}{label}_Dsmooth.obj")


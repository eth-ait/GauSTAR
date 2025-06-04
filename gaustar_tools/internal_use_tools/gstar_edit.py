import os
import cv2
import numpy as np
import trimesh
import torch
from tqdm import tqdm
import shutil
from scipy.spatial.transform import Rotation

from gaustar_utils.spherical_harmonics import (
    eval_sh, RGB2SH, SH2RGB,
)
from gaustar_tools.tracking_util import tracking_face
from gaustar_tools.warp_mesh import project, query_at_image


surface_triangle_bary_coords = torch.tensor(
                    [[2/3, 1/6, 1/6],
                    [1/6, 2/3, 1/6],
                    [1/6, 1/6, 2/3],
                    [1/6, 5/12, 5/12],
                    [5/12, 1/6, 5/12],
                    [5/12, 5/12, 1/6]],
                    dtype=torch.float32,
                )[..., None]


def best_fit_transform(A, B):
    '''
    FROM: https://github.com/ClayFlannigan/icp/blob/master/icp.py

    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def cut_gstar():
    work_dir = "//output/track_241111T2_SHreg/"
    edit_dir = "//output/track_241111T2_SHreg_cut_2/"

    for f_idx in tqdm(range(21, 22, 1)):
        # edit mesh
        mesh = trimesh.load_mesh(work_dir + f"{f_idx:04d}/color_mesh.obj")
        face_vert = mesh.vertices[mesh.faces]
        face_pose = np.mean(face_vert, axis=-2)

        face_mask = (face_pose[:, 0] > 1.2 * face_pose[:, 2] - 0.37)
        # face_mask = face_mask & (face_pose[:, 1] < 0.93)
        face_mask = face_mask & (face_pose[:, 0] < - 0.8 * face_pose[:, 2] + 0.6)
        face_mask = face_mask & (face_pose[:, 0] > - 0.8 * face_pose[:, 2] + 0.05)
        face_mask = face_mask & (face_pose[:, 1] < 0.35)

        face_mask_2 = (face_pose[:, 0] > 1.2 * face_pose[:, 2] - 0.37)
        face_mask_2 = face_mask_2 & (face_pose[:, 1] < 0.3)
        face_mask = face_mask | face_mask_2

        face_mask_3 = (face_pose[:, 0] < 1.2 * face_pose[:, 2] - 0.37)
        # face_mask_3 = face_mask_3 | (face_pose[:, 1] > 0.9)
        face_mask_3 = face_mask_3 & (face_pose[:, 0] < 0.3)
        face_mask = face_mask | face_mask_3

        face_mask_4 = (face_pose[:, 0] < 0.3) & (face_pose[:, 1] > 0.95)
        face_mask = face_mask | face_mask_4

        face_mask = ~face_mask
        mesh.update_faces(face_mask)
        mesh.remove_unreferenced_vertices()
        os.makedirs(edit_dir + f"{f_idx:04d}/", exist_ok=True)
        mesh.export(edit_dir + f"{f_idx:04d}/color_mesh.obj")

        # edit Gaussian
        gs_mask = face_mask.repeat(6)
        ckpt = torch.load(work_dir + f"{f_idx:04d}/2000.pt", map_location=torch.device('cpu'))

        ckpt['state_dict']['_surface_mesh_faces'] = torch.from_numpy(mesh.faces)
        ckpt['state_dict']['_points'] = torch.from_numpy(mesh.vertices)
        merge_keys = ['all_densities', '_scales', '_quaternions', '_delta_t', '_delta_r', '_sh_coordinates_dc', '_sh_coordinates_rest']
        for key in merge_keys:
            ckpt['state_dict'][key] = ckpt['state_dict'][key][gs_mask]

        ckpt.pop('optimizer_state_dict')
        torch.save(ckpt, edit_dir + f"{f_idx:04d}/2000.pt")


def merge_gstar():

    track_dir = "/mnt/euler/SUGAR/SuGaR/output/track_240724T12_SHreg/"
    dir_0 = "//output/track_240724T12_merge0/"
    dir_1 = "//output/track_240724T12_merge1/"
    out_dir = "//output/track_240724T12_merge1/"
    obj_file = dir_1 + "color_mesh.obj"
    ckpt_file = dir_1 + "2000.pt"

    trajectory_npz = np.load(track_dir + f"tracking/tracking_anchor_303points.npz")
    trajectory = trajectory_npz["face_trajectory"]
    track_frames = trajectory_npz["frames"]  # [frame_0, frame_end, interval]
    anchor_f0 = trajectory[0]

    for f_idx in tqdm(range(120, 300, 2)):
        cnt_idx = (f_idx - track_frames[0]) // track_frames[2]

        mesh_0 = trimesh.load_mesh(dir_0 + f"{f_idx:04d}/color_mesh.obj")
        mesh_1 = trimesh.load_mesh(obj_file)

        anchor_tracking = trajectory[cnt_idx]
        Rt, R, t = best_fit_transform(anchor_f0, anchor_tracking)

        mesh_1.vertices[..., 2] += 0.5
        mesh_1.vertices = np.matmul(R, mesh_1.vertices[..., None])[..., 0] + t

        if f_idx > 120:
            d_f = (f_idx - 120)
            if d_f > 60:
                d_f = 60
            degree = d_f * 0.45
            r1_mat = np.identity(3)
            r1_mat[:3, :3] = Rotation.from_euler("z", degree, degrees=True).as_matrix()

            mesh_center = mesh_1.vertices[10104]

            mesh_1.vertices = mesh_1.vertices - mesh_center
            mesh_1.vertices = np.matmul(r1_mat, mesh_1.vertices[..., None])[..., 0]
            mesh_1.vertices = mesh_1.vertices + mesh_center

            dist = d_f / 60 * np.asarray([0.04, 0.04, 0])
            mesh_1.vertices = mesh_1.vertices + dist

        vert_num_0 = mesh_0.vertices.shape[0]
        vert_merge = np.concatenate((mesh_0.vertices, mesh_1.vertices))
        face_merge = np.concatenate((mesh_0.faces, mesh_1.faces + vert_num_0))
        face_color_merge = np.concatenate((mesh_0.visual.face_colors, mesh_1.visual.face_colors))
        mesh_merge = trimesh.Trimesh(vertices=vert_merge, faces=face_merge, face_colors=face_color_merge, process=False)

        os.makedirs(out_dir + f"{f_idx:04d}", exist_ok=True)
        mesh_merge.export(out_dir + f"{f_idx:04d}/color_mesh.obj")

        ckpt_0 = torch.load(dir_0 + f"{f_idx:04d}/2000.pt", map_location=torch.device('cpu'))
        ckpt_1 = torch.load(ckpt_file, map_location=torch.device('cpu'))
        merge_keys = ['all_densities', '_scales', '_quaternions', '_delta_t', '_delta_r', '_sh_coordinates_dc', '_sh_coordinates_rest']
        ckpt_merge = ckpt_0.copy()
        ckpt_merge['state_dict']['_surface_mesh_faces'] = torch.from_numpy(face_merge)
        ckpt_merge['state_dict']['_points'] = torch.from_numpy(vert_merge)
        for key in merge_keys:
            ckpt_merge['state_dict'][key] = torch.cat((ckpt_0['state_dict'][key], ckpt_1['state_dict'][key]))

        ckpt_merge.pop('optimizer_state_dict')
        torch.save(ckpt_merge, out_dir + f"{f_idx:04d}/2000.pt")



def merge_gstar_robot():

    dir_0 = "//output/track_241111T2_SHreg_cut/"
    dir_1 = "//output/track_241111T2_SHreg_cut_2/"
    out_dir = "//output/track_241111T2_SHreg_merge/"
    obj_file = dir_1 + "0021/color_mesh.obj"
    ckpt_file = dir_1 + "0021/2000.pt"

    mesh_1 = trimesh.load_mesh(obj_file)
    ckpt_1 = torch.load(ckpt_file, map_location=torch.device('cpu'))

    for f_idx in tqdm(range(10, 170, 1)):

        mesh_0 = trimesh.load_mesh(dir_0 + f"{f_idx:04d}/color_mesh.obj")

        vert_num_0 = mesh_0.vertices.shape[0]

        noise = np.random.randn(*mesh_1.vertices.shape) * 0.0002

        vert_merge = np.concatenate((mesh_0.vertices, mesh_1.vertices + noise))
        face_merge = np.concatenate((mesh_0.faces, mesh_1.faces + vert_num_0))
        face_color_merge = np.concatenate((mesh_0.visual.face_colors, mesh_1.visual.face_colors))
        mesh_merge = trimesh.Trimesh(vertices=vert_merge, faces=face_merge, face_colors=face_color_merge, process=False)

        os.makedirs(out_dir + f"{f_idx:04d}", exist_ok=True)
        mesh_merge.export(out_dir + f"{f_idx:04d}/color_mesh.obj")

        ckpt_0 = torch.load(dir_0 + f"{f_idx:04d}/2000.pt", map_location=torch.device('cpu'))
        merge_keys = ['all_densities', '_scales', '_quaternions', '_delta_t', '_delta_r', '_sh_coordinates_dc', '_sh_coordinates_rest']
        ckpt_merge = ckpt_0.copy()
        ckpt_merge['state_dict']['_surface_mesh_faces'] = torch.from_numpy(face_merge)
        ckpt_merge['state_dict']['_points'] = torch.from_numpy(vert_merge)
        for key in merge_keys:
            ckpt_merge['state_dict'][key] = torch.cat((ckpt_0['state_dict'][key], ckpt_1['state_dict'][key]))

        # ckpt_merge.pop('optimizer_state_dict')
        torch.save(ckpt_merge, out_dir + f"{f_idx:04d}/2000.pt")


# def merge_gstar():
#
#     in_dir = "/media/dalco/data/SUGAR/SuGaR/output/track_241111T2_SHreg/"
#     out_dir = "/media/dalco/data/SUGAR/SuGaR/output/track_240724T12_merge1/"
#
#     for f_idx in tqdm(range(120, 300, 2)):
#         mesh = trimesh.load_mesh(in_dir + f"{f_idx:04d}/color_mesh.obj")
#
#         face_vert = mesh.vertices[mesh.faces]
#         face_pose = np.mean(face_vert, axis=-2)
#         face_mask = (face_pose[:, 1] > 0.71) & (face_pose[:, 1] < 0.75) & (face_pose[:, 2] > 0.6)
#
#         mesh.update_faces(face_mask)
#         mesh_cut = trimesh.Trimesh(vertices=vert_cut, faces=face_cut, face_colors=face_color_cut, process=False)
#
#         os.makedirs(out_dir + f"{f_idx:04d}", exist_ok=True)
#         mesh_cut.export(out_dir + f"{f_idx:04d}/color_mesh.obj")
#
#         ckpt_0 = torch.load(in_dir + f"{f_idx:04d}/2000.pt", map_location=torch.device('cpu'))
#         merge_keys = ['all_densities', '_scales', '_quaternions', '_delta_t', '_delta_r', '_sh_coordinates_dc', '_sh_coordinates_rest']
#         ckpt_merge = ckpt_0.copy()
#         ckpt_merge['state_dict']['_surface_mesh_faces'] = torch.from_numpy(face_merge)
#         ckpt_merge['state_dict']['_points'] = torch.from_numpy(vert_merge)
#         for key in merge_keys:
#             ckpt_merge['state_dict'][key] = torch.cat((ckpt_0['state_dict'][key], ckpt_1['state_dict'][key]))
#
#         torch.save(ckpt_merge, out_dir + f"{f_idx:04d}/2000.pt")



def select_tracking_point(work_dir):
    mesh = trimesh.load_mesh("/mnt/euler/SUGAR/SuGaR/output/track_240724T12_SHreg/0100/color_mesh.obj")
    face_vert = mesh.vertices[mesh.faces]
    face_pose = np.mean(face_vert, axis=-2)
    # face_mask = (face_pose[:, 1] > 1.76) & (face_pose[:, 1] < 1.8)  # 240724T12 hat tracking
    face_mask = (face_pose[:, 1] > 0.71) & (face_pose[:, 1] < 0.75) & (face_pose[:, 2] > 0.6)
    anchor_fid = np.where(face_mask)[0]

    # ckpt = torch.load(work_dir + f"{frame_0:04d}/2000.pt", map_location=torch.device('cpu'))
    # mesh_points = ckpt['state_dict']['_points']
    # mesh_faces = ckpt['state_dict']['_surface_mesh_faces']

    print(np.mean(face_pose[anchor_fid], axis=0))

    # anchor_fid = [95469, 90330, 95583, 90461]  # 0724T12
    # anchor_fid = [95476, 90485, 95586, 90695]
    # anchor_fid = [86464, 62473, 93274, 69837]  # 0906T3
    # anchor_fid = [88899, 68712, 94312, 73381]  # 0906T3
    # anchor_fid = [95134, 90600, 93015]
    anchor_num = len(anchor_fid)
    anchor_bary = np.ones((anchor_num, 3)) / 3

    os.makedirs(work_dir + "tracking/", exist_ok=True)
    np.savez_compressed(work_dir + f"tracking/anchor_{anchor_num}points.npz",
                        face_ids=np.array(anchor_fid), face_bary=np.array(anchor_bary))


def distance_point_to_face(face_vert, point):
    point_np = point.numpy()
    face_edge_0 = face_vert[1] - face_vert[0]
    face_edge_1 = face_vert[2] - face_vert[1]
    face_normal = -np.cross(face_edge_0, face_edge_1)
    face_normal = face_normal / np.linalg.norm(face_normal)
    point_edge = face_vert[0] - point_np
    point_dist = np.dot(point_edge, face_normal)
    return point_dist


def edit_gs_color(work_dir, edit_dir, frame_0, frame_end, interval, edit_name="cv2"):

    # cmr = np.load(data_dir + "rgb_cameras.npz")
    # intr = cmr["intrinsics"]
    # extr = cmr["extrinsics"]
    # shape = cmr["shape"]
    # cmr_num = shape.shape[0]

    os.makedirs(edit_dir, exist_ok=True)

    track_data = np.load(work_dir + "tracking/tracking_anchor_4points.npz")
    anchor_points_track = track_data["face_trajectory"]
    track_frames = track_data["frames"]  # [frame_0, frame_end, interval]

    for f_idx in tqdm(range(frame_0, frame_end, interval)):
        cnt_idx = (f_idx - track_frames[0]) // track_frames[2]

        ckpt = torch.load(work_dir + f"{f_idx:04d}/2000.pt", map_location=torch.device('cpu'))
        # base_mesh = trimesh.load_mesh(work_dir + f"{frame_0:04d}/color_mesh.obj")
        ckpt.pop('optimizer_state_dict')

        mesh_points = ckpt['state_dict']['_points']
        mesh_faces = ckpt['state_dict']['_surface_mesh_faces']
        faces_verts = mesh_points[mesh_faces]

        gs_points = faces_verts[:, None] * surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        gs_points = gs_points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        gs_points = gs_points.reshape(-1, 3)

        # if 1:
        #     cmr_idx = 12
        #     gs_pix = project(gs_points, intr[cmr_idx], extr[cmr_idx], shape[cmr_idx])
        #     edit_img = cv2.imread(edit_dir + f"{edit_name}.png")
        #     pix_color = query_at_image(edit_img, gs_pix)
        #     mask = (pix_color[..., 2] > 127) & (pix_color[..., 1] < 127)  # B-G-R
        #     average_scale = ckpt['state_dict']['_scales'].mean()
        #     # ckpt['state_dict']['_scales'][mask] = average_scale * 0.8
        #     ckpt['state_dict']['_scales'][...] = -6  # average_scale * 0.85  # (-6.1129)

        ckpt_rgb = SH2RGB(ckpt['state_dict']['_sh_coordinates_dc'])
        ckpt_scale = ckpt['state_dict']['_scales']

        texture_img = cv2.imread(edit_dir + f"{edit_name}.png", cv2.IMREAD_UNCHANGED)
        # texture_img = texture_img[360:1140, 190:870]  # thumb.png
        texture_img = texture_img[20:-20, 20:-20]  # cv2.png
        texture_shape = np.array(texture_img.shape[:2])

        # if 1:
        # anchor_vid = [47833, 45249, 47822, 45310]
        # anchor_points = mesh_points[anchor_vid]
        # anchor_face = mesh_faces[anchor_fid]
        # anchor_points = mesh_points[anchor_face]
        # anchor_points = np.average(anchor_points, axis=-2)
        anchor_points = anchor_points_track[cnt_idx]
        # anchor_UV = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        anchor_UV = np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])
        anchor_UV = np.concatenate((anchor_UV, np.zeros((anchor_UV.shape[0], 1))), axis=-1)
        anchor_triangle = [[0, 1, 2], [1, 2, 3]]

        for i in range(len(anchor_triangle)):
            anchor_face_vert = anchor_points[anchor_triangle[i]][None]
            # face_vert = face_vert.expand((gs_points.shape[0], 3, 3))  # tensor
            anchor_face_vert = np.broadcast_to(anchor_face_vert, (gs_points.shape[0], 3, 3))  # numpy
            gs_bary = trimesh.triangles.points_to_barycentric(anchor_face_vert, gs_points)
            gs_in_mask = np.all(gs_bary >= -0.03, axis=-1)

            gs_anchor_dist = distance_point_to_face(anchor_face_vert[0], gs_points)
            gs_in_mask = gs_in_mask & (-0.05 < gs_anchor_dist) & (gs_anchor_dist < 0.05)

            ckpt_scale[gs_in_mask] = -6

            face_UV = anchor_UV[anchor_triangle[i]]
            face_UV = np.broadcast_to(face_UV, (gs_in_mask.sum(), 3, 3))
            gs_UV = trimesh.triangles.barycentric_to_points(face_UV, gs_bary[gs_in_mask])[..., :2]
            gs_texture_pix = np.int32(gs_UV * texture_shape)
            gs_texture = texture_img[gs_texture_pix[:, 0], gs_texture_pix[:, 1]]

            # EMOJI
            # fg_mask = gs_texture[:, 3] > 50
            # in_ckpt_rgb = ckpt_rgb[gs_in_mask].clone()
            # in_ckpt_rgb[fg_mask] = torch.from_numpy(np.float32(gs_texture[fg_mask, None, :3] / 255))[..., [2, 1, 0]]
            # ckpt_rgb[gs_in_mask] = in_ckpt_rgb

            # LETTER
            ckpt_rgb[gs_in_mask] *= np.float32(gs_texture[:, 0, None, None] / 255)

        # mask = gs_points[..., 1] < 0.45
        # ckpt_rgb[mask] *= 0.1

        ckpt['state_dict']['_sh_coordinates_dc'] = RGB2SH(ckpt_rgb)
        ckpt['state_dict']['_scales'] = ckpt_scale

        os.makedirs(edit_dir + f"{f_idx:04d}/", exist_ok=True)
        torch.save(ckpt, edit_dir + f"{f_idx:04d}/2000.pt")

        shutil.copy(work_dir + f"{f_idx:04d}/color_mesh.obj", edit_dir + f"{f_idx:04d}/color_mesh.obj")

    # os.makedirs(edit_dir + "tracking/", exist_ok=True)
    # # np.save(edit_dir + f"tracking/{edit_name}.npy", gs_points[mask])
    # np.save(edit_dir + f"tracking/{edit_name}.npy", anchor_points)


def convert_mesh_to_gstar():
    mesh = trimesh.load_mesh("/media/dalco/data/SUGAR/obj/autumn_sword/ImageToStl.com_autumn_sword.obj")
    # mesh = trimesh.load_mesh("/media/dalco/data/SUGAR/obj/hat/clown_hat/ImageToStl.com_clown_hat.obj")
    mesh.visual = mesh.visual.to_color()

    mesh_center = np.average(mesh.bounds, axis=0)
    mesh_center_trans = np.identity(4)
    mesh_center_trans_inv = np.identity(4)
    mesh_center_trans[:3, 3] = -mesh_center
    mesh_center_trans_inv[:3, 3] = mesh_center

    vertices = mesh.vertices
    vertices_4 = np.concatenate((vertices, np.ones_like(vertices[:, 0])[..., None]), axis=-1)
    r1_mat = np.identity(4)
    r1_mat[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    r2_mat = np.identity(4)
    r2_mat[:3, :3] = Rotation.from_euler("y", -85, degrees=True).as_matrix()
    rt_mat = mesh_center_trans_inv @ r2_mat @ r1_mat @ mesh_center_trans
    vertices_4 = rt_mat @ vertices_4[..., None]
    vertices = vertices_4[:, :3, 0]

    # MEGUMI HAT
    # vertices = mesh.vertices * 0.25
    # vertices[:, 0] += -0.25
    # vertices[:, 1] += 1.95
    # vertices[:, 2] += 0.4
    # AUTUMN SWORD
    vertices = vertices * 0.07
    vertices[:, 0] += -0.3
    vertices[:, 1] += 0.55
    vertices[:, 2] += 0.1

    new_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, face_colors=mesh.visual.face_colors)
    new_mesh.export("/media/dalco/data/SUGAR/obj/autumn_sword/sword.obj")


def rot_mesh():
    mesh = trimesh.load_mesh("//output/track_240724T12_merge0/color_mesh.obj")
    vertices = mesh.vertices

    mesh_center = np.average(mesh.bounds, axis=0)
    mesh_center_trans = np.identity(4)
    mesh_center_trans_inv = np.identity(4)
    mesh_center_trans[:3, 3] = -mesh_center
    mesh_center_trans_inv[:3, 3] = mesh_center

    vertices_4 = np.concatenate((vertices, np.ones_like(vertices[:, 0])[..., None]), axis=-1)
    r_mat = np.identity(4)
    r_mat[:3, :3] = Rotation.from_euler("zyx", [-20, 0, 10], degrees=True).as_matrix()
    rt_mat = mesh_center_trans_inv @ r_mat @ mesh_center_trans
    vertices_4 = rt_mat @ vertices_4[..., None]
    vertices = vertices_4[:, :3, 0]

    vertices *= 0.8
    vertices[:, 0] += 0.07
    vertices[:, 1] += 0.46
    vertices[:, 2] += 0.2

    new_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, face_colors=mesh.visual.face_colors)
    new_mesh.export("/media/dalco/data/SUGAR/SuGaR/output/track_240724T12_merge0/hat_rot.obj")


if __name__ == "__main__":
    # work_dir = "/mnt/euler/SUGAR/SuGaR/output/track_240724T12_SHreg/"
    work_dir = "//output/track_240906T3_update_re/"
    edit_dir = "//output/track_240906T3_colorEdit0_re/"
    frame_0 = 20
    frame_end = 200
    interval = 1

    # select_tracking_point(work_dir)
    # edit_gs_color(work_dir, edit_dir, frame_0, frame_end, interval)
    # tracking_face()

    # convert_mesh_to_gstar()
    # rot_mesh()

    # cut_gstar()
    merge_gstar_robot()
    # merge_gstar()

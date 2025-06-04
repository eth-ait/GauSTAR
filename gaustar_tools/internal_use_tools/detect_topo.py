import numpy as np
import trimesh
from scene.dataset_readers import storePly
import torch
from tqdm import tqdm

def detect_by_gradient():
    root = "/media/dalco/data/SUGAR/SuGaR/gs_output/dress_00210_Take5/0095/"

    gs_0 = trimesh.load_mesh(root + "point_cloud/iteration_1/point_cloud.ply")
    pc_0 = np.array(gs_0.vertices)

    mesh = trimesh.load_mesh(f"//output/dress_00210T5_iso5k5k_100k/0095/color_mesh.obj")


    for i in range(500, 7500, 500):
        gs_1 = trimesh.load_mesh(root + f"point_cloud/iteration_{i}/point_cloud.ply")
        pc_1 = np.array(gs_1.vertices)

        diff = np.linalg.norm(pc_0 - pc_1, axis=1)
        color = (diff / 0.08) * 255
        color = np.clip(color, 0, 255)
        color = color[..., None].repeat(3, axis=1)
        storePly(root + f"point_cloud/diff_{i}.ply", pc_0, color)

        face_color = color.reshape((-1, 6, 3))
        face_color = np.average(face_color, axis=1)
        obj_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=face_color, process=False)
        obj_mesh.export(f"/media/dalco/data/SUGAR/SuGaR/output/dress_00210T5_iso5k5k_100k/0095/vis_mesh_{i}.obj")


def detect_by_appearance(root, f_idx, frame_0):
    cur_ckpt = torch.load(root + f"{f_idx:04d}/4000.pt", map_location='cpu')
    ref_ckpt = torch.load(root + f"{frame_0:04d}/4000.pt", map_location='cpu')
    cur_dc = cur_ckpt['state_dict']['_sh_coordinates_dc'].numpy()
    ref_dc = ref_ckpt['state_dict']['_sh_coordinates_dc'].numpy()

    dc_diff = np.abs(cur_dc - ref_dc).squeeze()
    face_color = dc_diff.reshape((-1, 6, 3))
    face_color = np.average(face_color, axis=1)
    face_color = np.minimum(face_color * 100, 255)

    mesh = trimesh.load_mesh(root + f"{f_idx:04d}/color_mesh.obj")
    out_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=face_color, process=False)
    out_mesh.export(root + f"{f_idx:04d}/dc_diff.obj")


if __name__ == "__main__":
    for i in tqdm(range(55, 65)):
        detect_by_appearance("/media/dalco/data/SUGAR/SuGaR/output/track_240522T4_SHReg1/", i, 20)
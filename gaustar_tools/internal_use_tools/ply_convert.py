import os
import trimesh
import numpy as np
import argparse
from tqdm import tqdm
from plyfile import PlyData
from gaussian_splatting.scene.dataset_readers import storePly

def ms2pc(path):

    textured_mesh = trimesh.load_mesh(path + "mesh-f00001.obj")
    textured_mesh.visual = textured_mesh.visual.to_color()

    pc_mesh = trimesh.load_mesh(path + "mesh_000000_smooth_300k.obj")
    # pc_mesh = trimesh.load_mesh(path + "points-f00001.ply")
    pc = np.array(pc_mesh.vertices)
    # pc_mesh_0 = trimesh.load_mesh(path + "mesh-f00001.ply")
    # pc = np.concatenate((np.array(pc_mesh.vertices), np.array(pc_mesh_0.vertices)), axis=0)

    _, _, closest_faces = trimesh.proximity.closest_point(textured_mesh, pc)
    pc_color = (textured_mesh.visual.face_colors[closest_faces])[:, 0:3]

    pc *= 0.001
    os.makedirs(path + "sparse/0/", exist_ok=True)
    storePly(path + "sparse/0/points3D.ply", pc, pc_color)
    print("store at: ", path + "sparse/0/points3D.ply")
    # add_bg(path, pc, pc_color)



def add_bg(path, fg_pc, fg_pc_color):
    bg_vert = PlyData.read(path + "bg_trained.ply")['vertex']
    bg_pc = np.vstack([bg_vert['x'], bg_vert['y'], bg_vert['z']]).T
    pc_number = bg_pc.shape[0]
    pc_color = np.broadcast_to(np.array([150, 180, 100]), (pc_number, 3))

    bg_pc[:, 1] *= -1
    bg_pc[:, 2] *= -1
    bg_pc -= np.array([-1.884568, 0.751817, -2.868025])
    bg_pc *= 0.65
    bg_pc += np.array([-0.187015, 2.101122, 0.062247])
    storePly(path + "bg_align.ply", bg_pc, pc_color)

    x_idx = (bg_pc[:, 0] > -0.8) & (bg_pc[:, 0] < 0.6)
    y_idx = (bg_pc[:, 1] > 0.25) & (bg_pc[:, 1] < 2.2)
    z_idx = (bg_pc[:, 2] > -0.5) & (bg_pc[:, 2] < 0.4)
    bg_idx = ~(x_idx & y_idx & z_idx)
    storePly(path + "bg_cut.ply", bg_pc[bg_idx], pc_color[bg_idx])

    full_pc = np.concatenate((fg_pc, bg_pc[bg_idx]))
    full_pc_color = np.concatenate((fg_pc_color, pc_color[bg_idx]))
    storePly(path + "points3D.ply", full_pc, full_pc_color)


def mesh_convert(path):
    # gt_mesh = trimesh.load_mesh("/media/dalco/data/humanrf/out/mocap_231020_t9F_LA8/results/mesh/" + "mesh_000001_smooth_300k.obj")
    gt_mesh = trimesh.load_mesh("/media/dalco/data/humanrf/out/mocap_231020_t9F_LA8/results/mesh/" + "mesh_000000.obj")
    vert = gt_mesh.vertices * 0.001

    coarse_mesh = trimesh.load_mesh(path + "sugarmesh_3Dgs7000_densityestim02_sdfnorm02_densify100499_0_entro7000_9000_level01_decim200000.ply")
    # coarse_mesh = trimesh.load_mesh("/media/dalco/data/humanrf/in/mocap_231020_Take9Full/frames_rename/" + "mesh-f00002.obj")
    # coarse_mesh.vertices *= 0.001
    # coarse_mesh.visual = coarse_mesh.visual.to_color()

    batch_size = 10000
    batch_num = vert.shape[0] // batch_size + 1
    closest_faces = []
    for i in tqdm(range(batch_num)):
        i0 = i * batch_size
        i1 = np.min((i0 + batch_size, vert.shape[0]))
        vert_batch = vert[i0:i1]
        _, _, closest_faces_batch = trimesh.proximity.closest_point(coarse_mesh, vert_batch)
        closest_faces.append(closest_faces_batch)
    closest_faces = np.concatenate(closest_faces)
    vert_color = (coarse_mesh.visual.face_colors[closest_faces])[:, 0:3]

    tmesh = trimesh.Trimesh(vertices=vert, faces=gt_mesh.faces, vertex_colors=vert_color)
    tmesh.export("/media/dalco/data/SUGAR/SuGaR/output/gt_3dhand_noise.obj")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    args = parser.parse_args()

    # ms2pc(args.path)
    # add_bg("/media/dalco/data/SUGAR/data/mocap/test_3dhand/")
    mesh_convert("//output/test_3dhand_den_low_1x/coarse_mesh/")

    print("--Done--")

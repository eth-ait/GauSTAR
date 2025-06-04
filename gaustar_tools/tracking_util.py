import numpy as np
import trimesh
import os
from tqdm import tqdm


def find_face_of_tags(base_dir, gt_dir, frame_0=0):
    kp_3d_dict = dict(np.load(gt_dir + "apriltag/keypoints-3d.npz", allow_pickle=True))
    tag_ids = kp_3d_dict['tag_ids']

    base_mesh = trimesh.load_mesh(base_dir + f"{frame_0:04d}/color_mesh.obj")
    mesh_face_position = np.average(base_mesh.vertices[base_mesh.faces], axis=-2)

    face_ids_list = []
    face_bary_list = []
    for t_idx in tag_ids:
        kp_3d = kp_3d_dict[t_idx].item()['keypoints_3d']
        kp_3d = kp_3d[0, :, :3]  # [frame, kp_idx, coordinate]
        for kp_id in range(5):
            face_pos_diff = kp_3d[kp_id] - mesh_face_position
            face_pos_diff = np.linalg.norm(face_pos_diff, axis=-1)
            face_idx = np.argmin(face_pos_diff)

            face_vert = base_mesh.vertices[None, base_mesh.faces[face_idx]]
            face_bary = trimesh.triangles.points_to_barycentric(face_vert, kp_3d[kp_id][None])
            face_ids_list.append(face_idx)
            face_bary_list.append(face_bary[0])

    os.makedirs(base_dir + "tracking/", exist_ok=True)
    np.savez_compressed(base_dir + "tracking/kp_id_bary.npz",
                        face_ids=np.array(face_ids_list), face_bary=np.array(face_bary_list))


def tracking_face(base_dir, tracking_file=None, frame_0=0, frame_end=0, interval=1, sample_points=True):
    out_dir = base_dir + "tracking/"
    os.makedirs(out_dir, exist_ok=True)
    debug = True

    # if not os.path.exists(tracking_dir + f"{frame_0:04d}/face_corr.npz"):
    mesh = trimesh.load_mesh(base_dir + f"{frame_0:04d}/color_mesh.obj")
    # else:
    #     mesh = trimesh.load_mesh(tracking_dir + f"{frame_0:04d}_update/color_mesh.obj")
    if sample_points:
        face_ids = np.arange(10, mesh.faces.shape[0], 200)
        # face_ids = np.arange(0, mesh.faces.shape[0], 400)
        face_ids = face_ids[:-1]

        face_ids_ori = face_ids.copy()
        face_track_num = face_ids.shape[0]
        face_bary = np.ones((face_track_num, 3)) / 3

    else:
        track_points_dict = dict(np.load(base_dir + f"tracking/{tracking_file}"))
        face_ids = track_points_dict['face_ids']
        face_ids_ori = face_ids.copy()
        face_track_num = face_ids.shape[0]
        face_bary = track_points_dict['face_bary']

    print(f"track {face_ids.shape[0]} points.")

    face_trajectory = []  # np.zeros((frame_num, face_track_num, 3))

    # frame_range = list(range(30, 120, 2)) + list(range(120, 140, 1)) + list(range(140, 300, 2))
    # for frame_i in tqdm(frame_range):
    for frame_i in tqdm(range(frame_0, frame_end, interval)):
        mesh = trimesh.load_mesh(base_dir + f"{frame_i:04d}/color_mesh.obj")

        # face_position2 = np.average(mesh.vertices[mesh.faces[face_ids]], axis=-2)
        # face_bary = trimesh.triangles.points_to_barycentric(mesh.vertices[mesh.faces[face_ids]], face_position)
        face_position = trimesh.triangles.barycentric_to_points(mesh.vertices[mesh.faces[face_ids]], face_bary)

        # No update
        if not os.path.exists(base_dir + f"{frame_i:04d}/face_corr.npz"):
            # if frame_i % 2 == 0:
            face_trajectory.append(face_position)
            continue

        # Load update info
        face_corr = np.load(base_dir + f"{frame_i:04d}/face_corr.npz")
        track_mask_all = face_corr['track_face_mask']
        face_track_mask = track_mask_all[face_ids]
        # if np.all(face_track_mask):
        #     face_pos_tracking[frame_i] = face_position
        #     continue

        new_mesh = trimesh.load_mesh(base_dir + f"{frame_i:04d}_update/color_mesh.obj")
        new_mesh_face_position = np.average(new_mesh.vertices[new_mesh.faces], axis=-2)

        for face_i in range(face_track_num):

            remain_face = False
            keep_bary_for_tracking = False
            if face_track_mask[face_i]:
                new_face_idx = np.sum(track_mask_all[:face_ids[face_i]])
                # assert np.linalg.norm(new_mesh_face_position[new_face_idx] - face_position[face_i]) < 0.001

                if keep_bary_for_tracking:
                    remain_face = True
                    face_ids[face_i] = new_face_idx

                else:
                    new_face_vert = new_mesh.vertices[None, new_mesh.faces[new_face_idx]]
                    new_face_bary = trimesh.triangles.points_to_barycentric(new_face_vert, face_position[None, face_i])
                    if np.all(new_face_bary >= 0):
                        remain_face = True
                        face_ids[face_i] = new_face_idx
                        face_bary[face_i] = new_face_bary

            if not remain_face:
                face_pos_diff = face_position[face_i] - new_mesh_face_position
                face_pos_diff = np.linalg.norm(face_pos_diff, axis=-1)
                new_face_idx = np.argmin(face_pos_diff)
                # face_position[face_i] = new_mesh_face_position[new_face_idx]

                face_ids[face_i] = new_face_idx
                new_face_vert = new_mesh.vertices[None, new_mesh.faces[new_face_idx]]
                new_face_bary = trimesh.triangles.points_to_barycentric(new_face_vert, face_position[None, face_i])
                if np.any(new_face_bary < 0):
                    new_face_bary = np.maximum(new_face_bary, 0)
                    new_face_bary = new_face_bary / new_face_bary.sum()
                face_bary[face_i] = new_face_bary

        # new_face_vert = new_mesh.vertices[new_mesh.faces[face_ids]]
        # face_bary = trimesh.triangles.points_to_barycentric(new_face_vert, face_position)
        assert np.all(face_bary >= 0)

        # if frame_i % 2 == 0:
        face_trajectory.append(face_position)

        if debug and frame_i != frame_0:
            face_movement = face_trajectory[-1] - face_trajectory[-2]
            face_movement = np.linalg.norm(face_movement, axis=-1)
            if np.any(face_movement > 0.2):
                print(face_movement)
                # input()

    face_trajectory = np.array(face_trajectory)
    debug_ply = trimesh.points.PointCloud(face_trajectory.reshape(-1, 3))
    debug_ply.export(out_dir + f"tracking_debug.ply")

    if tracking_file and not sample_points:
        out_file_name = "tracking_" + tracking_file
    else:
        out_file_name = f"tracking_{face_track_num}faces_{frame_0}to{frame_end}.npz"
    np.savez_compressed(out_dir + out_file_name,
                        face_ids_ori=face_ids_ori,
                        frames=np.array([frame_0, frame_end, interval]),
                        face_trajectory=face_trajectory)


if __name__ == "__main__":
    base_dir = "/mnt/server02/GSTAR/track_241028T4/"
    gt_dir = "/media/dalco/Data_Chengwei/humanrf/in/mocap_241016_Take4/"  # ONLY for gt Apriltags
    frame_0 = 0
    frame_end = 229
    interval = 1

    # find_face_of_tags(base_dir, gt_dir, frame_0)
    tracking_face(base_dir, "kp_id_bary.npz", frame_0, frame_end, interval, sample_points=True)
    # mesh_edit()

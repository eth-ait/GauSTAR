import numpy as np
import os
import cv2
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation
from pathlib import Path
from plyfile import PlyData, PlyElement
from render_depth_from_mesh import render_mesh_depth_w_aitviewer
from actorshq.dataset.camera_data import read_calibration_csv


def cmr_convert(ahq_dir, gstar_dir):
    camera_csv = read_calibration_csv(Path(ahq_dir) / "calibration.csv")
    cmr_num = len(camera_csv)

    extr = np.zeros((cmr_num, 3, 4))
    intr = np.zeros((cmr_num, 3, 3))
    dist_coeffs = np.zeros((cmr_num, 5))
    shape = np.zeros((cmr_num, 2), dtype=np.int32)
    cmr_ids = list(range(1, 1+cmr_num))

    for i in range(cmr_num):
        r_mat = Rotation.from_rotvec(camera_csv[i].rotation_axisangle).as_matrix().T
        t_vec = -r_mat @ camera_csv[i].translation
        extr[i, :, :3] = r_mat
        extr[i, :, 3] = t_vec

        fx = camera_csv[i].focal_length[0] * camera_csv[i].width
        fy = camera_csv[i].focal_length[1] * camera_csv[i].height
        intr[i, 0, 0] = fx
        intr[i, 1, 1] = fy
        intr[i, 0, 2] = camera_csv[i].cx_pixel
        intr[i, 1, 2] = camera_csv[i].cy_pixel
        intr[i, 2, 2] = 1

        shape[i, 0] = camera_csv[i].height
        shape[i, 1] = camera_csv[i].width

    os.makedirs(gstar_dir, exist_ok=True)
    np.savez_compressed(gstar_dir / "rgb_cameras.npz",
                        ids=cmr_ids,
                        intrinsics=intr,
                        extrinsics=extr,
                        dist_coeffs=dist_coeffs,
                        shape=shape,
                        )


def img_translate(img, intr_mat, border_value=None):
    shape = img.shape[:2]
    dx = intr_mat[0, 2] - 0.5 * shape[1]  # col
    dy = intr_mat[1, 2] - 0.5 * shape[0]  # row
    trans = np.float32([[1, 0, -dx], [0, 1, -dy]])
    if border_value:
        img_trans = cv2.warpAffine(img, trans, shape[::-1], borderValue=border_value)
    else:
        img_trans = cv2.warpAffine(img, trans, shape[::-1])
    return img_trans


def rgb_convert(ahq_dir, gstar_dir, frame_0, frame_end, interval=1):
    camera_infos = dict(np.load(gstar_dir / "rgb_cameras.npz"))
    intr = camera_infos["intrinsics"]
    cmr_num = intr.shape[0]

    for f_idx in tqdm(range(frame_0, frame_end, interval)):

        frame_gstar_dir = gstar_dir / f"{f_idx:04d}/"
        rgb_dir = frame_gstar_dir / f"images/"
        os.makedirs(rgb_dir, exist_ok=True)
        # mask_dir = frame_gstar_dir / f"masks/"
        # os.makedirs(mask_dir, exist_ok=True)
        # depth_dir = frame_gstar_dir / f"depth/"
        # os.makedirs(depth_dir, exist_ok=True)

        for cmr_idx in range(cmr_num):
            cmr_name = f"Cam{(cmr_idx+1):03d}"
            rgb_ahq = cv2.imread(str(ahq_dir / f"rgbs/{cmr_name}/{cmr_name}_rgb{f_idx:06d}.jpg"), cv2.IMREAD_UNCHANGED)
            rgb_trans = img_translate(rgb_ahq, intr[cmr_idx])
            cv2.imwrite(str(rgb_dir / f"img_{cmr_idx:04d}.jpg"), rgb_trans)


def points_to_local_points(points: np.ndarray, extr):
    rot = extr[:3, :3]
    trans = extr[:3, 3, None]
    local_points = (np.matmul(rot, points.T) + trans).T
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
        return pixels.reshape((*batch_shape, 2)), local_points
    else:
        return pixels.reshape((*batch_shape, 2))


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


def compute_mesh_color(mesh_dir, gstar_dir, frame_0, label="_humanrf", res="100k", debug=False):
    camera_infos = dict(np.load(gstar_dir / "rgb_cameras.npz"))
    intr = camera_infos["intrinsics"]
    extr = camera_infos["extrinsics"]
    shape = camera_infos["shape"]
    cmr_num = intr.shape[0]

    mesh = trimesh.load_mesh(mesh_dir / f"mesh_{frame_0:06d}_smooth_{res}.obj")
    vert_num = mesh.vertices.shape[0]
    vert_color_total = np.zeros((cmr_num, vert_num, 3))
    vert_visual_total = np.zeros((cmr_num, vert_num))

    frame_dir = gstar_dir / f"{frame_0:04d}/"
    if debug:
        os.makedirs(frame_dir / "debug/", exist_ok=True)

    for cmr_idx in range(cmr_num):
        pix_pos, local_points = project(mesh.vertices, intr[cmr_idx], extr[cmr_idx], shape[cmr_idx], True)
        rgb_img = cv2.imread(frame_dir / f"images/img_{cmr_idx:04d}.jpg")
        vert_color = query_at_image(rgb_img, pix_pos)[..., ::-1]

        depth_img = np.load(frame_dir / f"depth{label}/img_{cmr_idx:04d}_depth.npz")['depth']
        pix_depth, valid_mask = query_at_image(depth_img, pix_pos, return_valid=True)

        depth_diff_cur = np.abs(local_points[..., 2] - pix_depth)
        visual_mask = valid_mask & (depth_diff_cur < 0.005)
        vert_color_total[cmr_idx][visual_mask] = vert_color[visual_mask]
        vert_visual_total[cmr_idx] = visual_mask

        if debug:
            visual_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vert_color_total[cmr_idx])
            visual_mesh.export(frame_dir / f"debug/visual_{cmr_idx:04d}.obj")

    vert_visual_cnt = np.sum(np.int32(vert_visual_total), axis=0)[..., None]
    vert_color = np.sum(vert_color_total, axis=0) / vert_visual_cnt
    color_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vert_color)
    color_mesh.export(gstar_dir / f"init_mesh_{res}.obj")


def write_colmap_cameras_text(intr, shape, path):
    cmr_number = intr.shape[0]
    HEADER = "# Camera list with one line of data per camera:\n" + \
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy]\n" + \
             "# Number of cameras: {}\n".format(cmr_number)
    with open(path, "w") as fout:
        fout.write(HEADER)
        for i in range(cmr_number):
            intr_mat = intr[i]
            to_write = [i, 'PINHOLE', int(shape[i, 1]), int(shape[i, 0]),
                        intr_mat[0, 0], intr_mat[1, 1], shape[i, 1] * 0.5, shape[i, 0] * 0.5]  # intr_mat[0, 2], intr_mat[1, 2]
            line = " ".join([str(elem) for elem in to_write])
            fout.write(line + "\n")


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_colmap_images_text(images, path):
    HEADER = "# Image list with two lines of data per image:\n" + \
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n" + \
             "# Number of images: {}, mean observations per image: {}\n".format(len(images), 0)
    with open(path, "w") as fout:
        fout.write(HEADER)
        for i in range(images.shape[0]):
            extr = images[i]
            R_vec = rotmat2qvec(extr[0:3, 0:3])
            T_vec = extr[0:3, 3]
            image_header = [i, *R_vec, *T_vec, i, f"img_{i:04d}.jpg"]
            first_line = " ".join(map(str, image_header))
            fout.write(first_line + "\n")

            points_strings = []
            fout.write(" ".join(points_strings) + "\n")


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def colmap_convert(gstar_dir, frame_0, res="100k"):
    camera_infos = dict(np.load(gstar_dir / "rgb_cameras.npz"))
    intr = camera_infos["intrinsics"]
    extr = camera_infos["extrinsics"]
    shape = camera_infos["shape"]

    colmap_dir = gstar_dir / f"{frame_0:04d}/sparse/0/"
    os.makedirs(colmap_dir, exist_ok=True)

    write_colmap_cameras_text(intr, shape, colmap_dir / "cameras.txt")
    write_colmap_images_text(extr, colmap_dir / "images.txt")

    mesh = trimesh.load_mesh(gstar_dir / f"init_mesh_{res}.obj")
    # pc = trimesh.points.PointCloud(mesh.vertices, colors=mesh.visual.vertex_colors)
    # pc.export(colmap_dir / "points3D.ply", file_type='ply', encoding='ascii')
    storePly(colmap_dir / "points3D.ply", mesh.vertices, mesh.visual.vertex_colors[..., :3])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaustar_data_dir", required=True)
    parser.add_argument("--actorshq_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--frame_0", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=0)
    args = parser.parse_args()

    gstar_dir = Path(args.gaustar_data_dir)
    ahq_dir = Path(args.actorshq_dir)
    mesh_dir = Path(args.mesh_dir)
    frame_0 = args.frame_0
    frame_end = args.frame_end

    # Convert ActorHQ camera to GauSTAR camera
    cmr_convert(ahq_dir, gstar_dir)

    # Convert ActorHQ rgb images to GauSTAR rgb images
    print("Converting images...")
    rgb_convert(ahq_dir, gstar_dir, frame_0, frame_end)

    # Render depth and mask for GauSTAR using HumanRF meshes
    print("Rendering depth...")
    render_mesh_depth_w_aitviewer(
        camera_path=gstar_dir / "rgb_cameras.npz",
        mesh_folder=mesh_dir,
        gstar_folder=gstar_dir,
        wo_cxcy=True,
        from_humanrf=True,
        frame_0=frame_0,
        frame_end=frame_end,
        interval=1,
    )

    # Get mesh for the first frame
    compute_mesh_color(mesh_dir, gstar_dir, frame_0)

    # Convert to COLMAP dataset
    colmap_convert(gstar_dir, frame_0)

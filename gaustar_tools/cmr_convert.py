import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
import trimesh
from gaussian_splatting.scene.dataset_readers import storePly

# from tifffile import imread, imwrite
# import tifffile as tf
import pyfqmr
import glob
import math
import shutil

def write_cameras_text(intr, shape, path):
    cmr_number = intr.shape[0]
    HEADER = "# Camera list with one line of data per camera:\n" + \
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy]\n" + \
             "# Number of cameras: {}\n".format(cmr_number)
    with open(path, "w") as fout:
        fout.write(HEADER)
        for i in range(cmr_number):
            intr_mat = intr[i]
            to_write = [i, 'PINHOLE', shape[i, 1], shape[i, 0],
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


def write_images_text(images, path):
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


def cmr_convert(path, intr, extr, shape):
    colmap_dir = path + "sparse/0/"
    os.makedirs(colmap_dir, exist_ok=True)
    write_cameras_text(intr, shape, colmap_dir + "cameras.txt")
    write_images_text(extr, colmap_dir + "images.txt")


def img_convert_dfa(path, intr, shape, for_rgb=True, for_mask=True):
    if for_rgb:
        rgb_dir = path + "images/"
        os.makedirs(rgb_dir, exist_ok=True)
    if for_mask:
        mask_dir = path + "masks/"
        os.makedirs(mask_dir, exist_ok=True)

    # max_margin = 0

    # cmr_number = intr.shape[0]
    for i in tqdm(range(intr.shape[0])):
        intr_mat = intr[i]
        dx = intr_mat[0, 2] - 0.5 * shape[i, 1]  # col
        dy = intr_mat[1, 2] - 0.5 * shape[i, 0]  # row
        trans = np.float32([[1, 0, -dx], [0, 1, -dy]])
        # max_margin = np.max([max_margin, np.abs(dx), np.abs(dy)])

        if for_rgb:
            rgb_mocap = cv2.imread(path + f"images_mocap/img_{i:04d}.jpg", cv2.IMREAD_UNCHANGED)
            rgb_trans = cv2.warpAffine(rgb_mocap, trans, shape[i, ::-1])
            cv2.imwrite(rgb_dir + f"img_{i:04d}.png", rgb_trans)
        if for_mask:
            mask_mocap = cv2.imread(path + f"masks_mocap/img_{i:04d}_alpha.png", cv2.IMREAD_UNCHANGED)
            mask_trans = cv2.warpAffine(mask_mocap, trans, shape[i, ::-1])
            cv2.imwrite(mask_dir + f"img_{i:04d}_alpha.png", mask_trans)
    # print("max margin: ", max_margin)


def img_convert(img, intr_mat, border_value=None):
    shape = img.shape[:2]
    dx = intr_mat[0, 2] - 0.5 * shape[1]  # col
    dy = intr_mat[1, 2] - 0.5 * shape[0]  # row
    trans = np.float32([[1, 0, -dx], [0, 1, -dy]])
    if border_value:
        img_trans = cv2.warpAffine(img, trans, shape[::-1], borderValue=border_value)
    else:
        img_trans = cv2.warpAffine(img, trans, shape[::-1])
    return img_trans


def add_color_to_pc(textured_mesh, pc):
    textured_mesh.visual = textured_mesh.visual.to_color()
    batch_size = 50000
    # batch_num = pc.shape[0] // batch_size + 1
    batch_num = math.ceil(pc.shape[0] / batch_size)
    closest_faces = []
    for i in range(batch_num):
    # for i in tqdm(range(batch_num)):
        i0 = i * batch_size
        i1 = np.min((i0 + batch_size, pc.shape[0]))
        vert_batch = pc[i0:i1]
        _, _, closest_faces_batch = trimesh.proximity.closest_point(textured_mesh, vert_batch)
        closest_faces.append(closest_faces_batch)
    closest_faces = np.concatenate(closest_faces)
    pc_color = (textured_mesh.visual.face_colors[closest_faces])[:, 0:3]
    return pc_color


def rename_MS_mesh(path, upper=True):
    texture = sorted(glob.glob(f"{path}/*.png"))
    for name in texture:
        file = os.path.basename(name)
        if upper:
            new_name = path + f"/Atlas-F{file[7:12]}.png"
        else:
            new_name = path + f"/atlas-f{file[7:12]}.png"
        os.rename(name, new_name)

    matlib = sorted(glob.glob(f"{path}/*.mtl"))
    for name in matlib:
        file = os.path.basename(name)
        if upper:
            new_name = path + f"/MatLib-F{file[8:13]}.mtl"
        else:
            new_name = path + f"/matlib-f{file[8:13]}.mtl"
        os.rename(name, new_name)


def dataset_seq_convert(root, out_dir, frame_0=0, frame_end=0, interval=1):
    '''
    Root: should have
        - cmr_file as rgb_cameras.npz
        - image in img_folder (.jpg)
        - undistorted mask in mask_folder (.png)
        - texture obj mesh in texture_mesh_folder (.obj & texture)
        - point cloud in pc_folder (.ply)
    Output:
        - images for GS
        - masks for GS
        - COLMAP data for GS
        - (optional) coarse mesh for SUGAR

    '''

    cmr_file = root + "cameras/rgb_cameras.npz"
    img_folder = root + "images/"
    # mask_folder = root + "scan_mask_undistort_wocxcy/"
    mask_folder = root + "scan_mask_undistort/"
    depth_folder = root + "scan_depth_undistort_wocxcy/"
    # depth_folder = root + "dpt/"
    texture_mesh_folder = root + "frames/"
    pc_folder = root + "smoothmeshes/"


    cmr = np.load(cmr_file)
    intr = cmr["intrinsics"]
    extr = cmr["extrinsics"]
    dist_coeff = cmr["dist_coeffs"]
    shape = cmr["shape"]
    cmr_rename = cmr["ids"]
    # np.savez_compressed(out_dir + "rgb_cameras.npz", cmr)
    shutil.copyfile(cmr_file, out_dir + "rgb_cameras.npz")

    # frame_0 = 10
    # frame_num = 59
    cmr_num = intr.shape[0] # - 5
    for f_idx in tqdm(range(frame_0, frame_end, interval)):

        frame_out_dir = out_dir + f"{f_idx:04d}/"
        if f_idx == frame_0:  # GS input data
            # CAMERA PARAMETERS
            cmr_convert(frame_out_dir, intr, extr, shape)

            # POINT CLOUD
            rename_MS_mesh(texture_mesh_folder)
            textured_mesh = trimesh.load_mesh(texture_mesh_folder + f"mesh-f{(f_idx+1):05d}.obj")
            pc_mesh = trimesh.load_mesh(pc_folder + f"mesh-f{(f_idx+1):05d}.ply")
            # pc_mesh = trimesh.load_mesh("/media/dalco/data/SUGAR/SuGaR/output/dress_00210T5_iso5k5k_100k/0095/0095.ply")
            pc = np.array(pc_mesh.vertices)
            pc_color = add_color_to_pc(textured_mesh, pc)
            pc *= 0.001
            storePly(frame_out_dir + "sparse/0/points3D.ply", pc, pc_color)
            rename_MS_mesh(texture_mesh_folder, upper=False)

        # IMAGES and MASKS
        rgb_dir = frame_out_dir + f"images/"
        os.makedirs(rgb_dir, exist_ok=True)
        mask_dir = frame_out_dir + f"masks/"
        os.makedirs(mask_dir, exist_ok=True)
        depth_dir = frame_out_dir + f"depth/"
        os.makedirs(depth_dir, exist_ok=True)

        for cmr_idx in tqdm(range(cmr_num)):

            cmr_name = cmr_rename[cmr_idx]

            # RGB IMAGES
            rgb_mocap = cv2.imread(img_folder + f"{cmr_name}/{(f_idx+1):06d}.jpg", cv2.IMREAD_UNCHANGED)

            rgb_mocap = cv2.undistort(rgb_mocap, intr[cmr_idx], dist_coeff[cmr_idx])
            rgb_trans = img_convert(rgb_mocap, intr[cmr_idx])
            cv2.imwrite(rgb_dir + f"/img_{cmr_idx:04d}.jpg", rgb_trans)

            # MASK
            # mask_mocap = cv2.imread(mask_folder + f"{cmr_name}/{(f_idx+1):06d}.png", cv2.IMREAD_UNCHANGED)
            # if mask_mocap.shape[-1] == 3:
            #     mask_mocap = np.uint8((mask_mocap[..., 0] > 100) * 255)
            # mask_trans = img_convert(mask_mocap, intr[cmr_idx])
            # # mask_trans = mask_mocap
            # cv2.imwrite(mask_dir + f"/img_{cmr_idx:04d}_alpha.png", mask_trans)

            # DEPTH
            # if os.path.exists(depth_folder + f"{cmr_name}/{(f_idx+1):06d}.tiff"):
            #     depth_mocap = cv2.imread(depth_folder + f"{cmr_name}/{(f_idx+1):06d}.tiff", cv2.IMREAD_UNCHANGED)
            # elif os.path.exists(depth_folder + f"{cmr_name}/{(f_idx+1):06d}.npz"):
            #     depth_mocap = np.load(depth_folder + f"{cmr_name}/{(f_idx+1):06d}.npz")['depth']
            # else:
            #     raise RuntimeError("No depth data found")
            # depth_trans = depth_mocap  # img_convert(depth_mocap, intr[cmr_idx], border_value=depth_mocap.max().item())
            #
            # # tf.imwrite(depth_dir + f"/img_{cmr_idx:04d}_depth.tiff", depth_trans, compression='zlib')
            # np.savez_compressed(depth_dir + f"/img_{cmr_idx:04d}_depth.npz", depth=depth_trans)
            # depth_vis = np.minimum(10, np.array(depth_trans))
            # depth_vis = np.uint8(depth_vis / 10 * 255.0)
            # depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            # cv2.imwrite(depth_dir + f"/img_{cmr_idx:04d}_depth.jpg", depth_vis)


def mesh_seq_convert(root, mesh_out_dir, frame_0=0, triangle_num=None, sub_dir=False):
    texture_mesh_folder = root + "frames/"
    pc_folder = root + "smoothmeshes/"

    rename_MS_mesh(texture_mesh_folder)
    # mesh_out_dir = out_dir + ""

    # frame_0 = 70
    frame_num = 1
    for f_idx in tqdm(range(frame_0, frame_0 + frame_num)):
        textured_mesh = trimesh.load_mesh(texture_mesh_folder + f"mesh-f{(f_idx + 1):05d}.obj")
        pc_mesh = trimesh.load_mesh(pc_folder + f"mesh-f{(f_idx + 1):05d}.ply")
        if triangle_num is not None:
            mesh_simplifier = pyfqmr.Simplify()
            mesh_simplifier.setMesh(pc_mesh.vertices, pc_mesh.faces)
            mesh_simplifier.simplify_mesh(target_count=triangle_num)
            pc, faces, _ = mesh_simplifier.getMesh()
        else:
            pc = np.array(pc_mesh.vertices)
            faces = pc_mesh.faces
        pc_color = add_color_to_pc(textured_mesh, pc)
        pc *= 0.001
        coarse_mesh = trimesh.Trimesh(vertices=pc, faces=faces, vertex_colors=pc_color)

        if sub_dir:
            mesh_out_dir = mesh_out_dir + f"{f_idx:04d}/"
        os.makedirs(mesh_out_dir, exist_ok=True)
        coarse_mesh.export(mesh_out_dir + f"MS_{int(triangle_num / 1000)}k.obj", file_type="obj")

    rename_MS_mesh(texture_mesh_folder, upper=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--frame_0", required=True, type=int)
    parser.add_argument("--frame_end", required=True, type=int)
    parser.add_argument("--interval", default=1, type=int)
    args = parser.parse_args()

    mesh_seq_convert(args.in_dir, args.out_dir, args.frame_0, triangle_num=100000)
    dataset_seq_convert(args.in_dir, args.out_dir, args.frame_0, args.frame_end, args.interval)
    # dataset_seq_convert("/media/dalco/data/humanrf/in/mocap_00210_Take5/", "/media/dalco/data/SUGAR/data/mocap/dress_00210_Take5/")

    exit(0)

    cmr = np.load(args.path + "rgb_cameras.npz")
    intr = cmr["intrinsics"]
    extr = cmr["extrinsics"]
    shape = cmr["shape"]

    cmr_convert(args.path, intr, extr, shape)
    img_convert_dfa(args.path, intr, shape, for_rgb=True, for_mask=True)

    print("--Done--")

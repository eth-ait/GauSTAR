import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def create_alpha_channel():
    for i in range(1, 4):
        img = cv2.imread(f"//output/track_240906T3_update_re/render/16/{i:04d}.jpg")
        bg_mask = (img[..., 0] > 245) & (img[..., 1] > 245) & (img[..., 2] > 245)
        bg_mask = (~bg_mask * 255)[..., None]
        rgba = np.concatenate((img, bg_mask), axis=-1)
        cv2.imwrite(f"//output/track_240906T3_update_re/render/16/{i:04d}_245.png", rgba)


def merge_appearance_geometry_imgs():
    root = "/media/dalco/data/SUGAR/fig/rebuttal/"
    file = "17_0070"
    img_a = cv2.imread(root + f"{file}_a.jpg")
    img_g = cv2.imread(root + f"{file}_g.jpg")
    height, width = img_g.shape[:2]
    img_mask = np.ones((height, width), dtype=bool)
    for row in range(height):
        for col in range(width):
            # if (row - 3650) > -2 * (col - 545):  # pipeline
            if (row - 2000) > -1 * (col - 1600):  # hi4d
            # if (row - 2000) > -0.8 * (col - 1400):  # 0906T8
            # if (row - 3050) > -0.9 * (col - 1800):  # robot
            # if (row - 3150) > -1.5 * (col - 1000):  # 0906T8
            # if (row - 2050) > -0.8 * (col - 1450):  # 1213T8
            # if (row - 200) > -1 * (col - 420):  # pano
            # if (row - 560) > -0.9 * (col - 350):  # ahq
                img_mask[row, col] = False

    img_ag = img_g
    img_ag[img_mask] = img_a[img_mask]
    # img_ag = img_ag[1800:, :1700]
    cv2.imwrite(root + f"{file}_ag.jpg", img_ag)


def cut_1_img():
    img = cv2.imread(f"/media/dalco/Data_Chengwei_21/SuGaR/data/track_240724_Take12/0129/images/img_0014.jpg")
    img = img[1800:, :1700]
    cv2.imwrite(f"//output/track_240724T12_figure/fig/input.jpg", img)


def cut_img():
    for i in tqdm(range(10, 170)):
        img = cv2.imread(f"//output/track_241111T2_SHreg_merge/tracking/tracking_499faces_10to170/16_{i:04d}.jpg")
        img[3115:3250, 1915:1990] = 255
        img[3220:3250, 1980:2020] = 255
        cv2.imwrite(f"//output/track_241111T2_SHreg_merge/tracking/tracking_499faces_10to170_cut/16_{i:04d}.jpg", img)


def cut_img_by_mask():
    mask = cv2.imread(f"/media/dalco/Data_Chengwei_21/SuGaR/data/track_241111_Take2/0010/masks/img_0016_alpha.png")
    r0 = 2250
    r1 = 2420
    c0 = 1850
    c1 = 2050
    local_mask = (mask[r0:r1, c0:c1] < 100)
    for i in tqdm(range(10, 170)):
        img = cv2.imread(f"//output/track_241111T2_SHreg_merge/{i:04d}/render_w/render_000016.jpg")
        local_img = img[r0:r1, c0:c1]
        local_img[local_mask] = 255
        img[r0:r1, c0:c1] = local_img
        cv2.imwrite(f"//output/track_241111T2_SHreg_merge/{i:04d}/render_w/render_000016_cut.jpg", img)


def overlap_img():
    img_fg = cv2.imread(f"/media/dalco/data/SUGAR/fig/teaser/render/highlight/new_face2.png")
    fg_mask = img_fg[..., 1] < 150
    img_bg = cv2.imread(f"/media/dalco/data/SUGAR/fig/teaser/render/color_mesh.png")
    img_bg[fg_mask] = img_fg[fg_mask]
    cv2.imwrite(f"/media/dalco/data/SUGAR/fig/teaser/render/overlap.png", img_bg)


def rot_mesh():
    root = "/media/dalco/data/SUGAR/SuGaR/output/track_240724T12_update_floor/"
    i = 129
    mesh = trimesh.load_mesh(root + f"{i:04d}/color_mesh.obj")

    r_mat = np.identity(3)
    r_mat[:3, :3] = Rotation.from_euler("zyx", [0, -60, 0], degrees=True).as_matrix()
    rot_vertices = r_mat @ mesh.vertices[..., None]
    rot_vertices = rot_vertices[..., 0]

    trimesh.Trimesh(vertices=rot_vertices, faces=mesh.faces, face_colors=mesh.visual.face_colors).export(root + f"{i:04d}/color_mesh_rot60.obj")


def show_unbind_weight():
    img = cv2.imread("/media/dalco/data/SUGAR/fig/img_0000_w.jpg")
    img = np.uint8(img)
    img_cm = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    bg_img = cv2.imread("/media/dalco/data/SUGAR/fig/img_0000_alpha.png")
    bg_mask = bg_img[..., 0] < 50

    img_cm[bg_mask] = 255
    cv2.imwrite("/media/dalco/data/SUGAR/fig/img_cm.jpg", img_cm)

    # mesh = trimesh.load_mesh("/media/dalco/data/SUGAR/fig/depth_diff_voxel_fc.obj")
    # # mesh.visual = mesh.visual.to_color()
    # vertex_colors = mesh.visual.vertex_colors
    #
    # # face_vert = mesh.vertices[mesh.faces]
    # # face_pose = np.mean(face_vert, axis=-2)
    # vert_mask = (mesh.vertices[:, 1] > 0.41) & (mesh.vertices[:, 1] < 0.44) & (mesh.vertices[:, 2] > 0.5)
    # vertex_colors = np.float32(vertex_colors)
    # vertex_colors[vert_mask] = np.minimum(vertex_colors[vert_mask] * 1.5, 255)
    # vertex_colors = np.uint8(vertex_colors)
    #
    # new_mesh = (trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors))
    # new_mesh.export("/media/dalco/data/SUGAR/fig/unbind_weight.obj")


# overlap_img()
# show_unbind_weight()
# cut_img()
# cut_img_by_mask()
merge_appearance_geometry_imgs()
# rot_mesh()

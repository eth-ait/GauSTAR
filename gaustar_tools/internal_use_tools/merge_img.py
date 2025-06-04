import cv2
import os
import numpy as np
from tqdm import tqdm

frame_0 = 0
frame_end = 200
interval = 1
pix_pad = (500, 482, 160, 500)  # top, bottom, left, right
# pix_pad = (900, 82, 350, 730)  # top, bottom, left, right
# pix_pad = (1000, 100, 100, 700)
resize = True

# width:  3004 - 1050 = 1954  or  3004 - 660 = 1954 * 1.2  or  3004 - 464 = 1954 * 1.3
# height: 4092 - 1500 = 2592  or  4092 - 982 = 2592 * 1.2  or  4092 - 722 = 2592 * 1.3

# width:  3004 - 1400 = 1604  or  3004 - 1080 = 1604 * 1.2  or 3004 - 1720 = 1604 * 0.8
# height: 4092 - 1500 = 2592  or  4092 - 982 = 2592 * 1.2   or 4092 - 2018 = 2592 * 0.8

# in_dir = "/media/dalco/Data_Chengwei_21/SuGaR/data/track_240724_Take12/"
in_dir = "/mnt/euler/SUGAR/data/mocap/track_241213_Take8/"
out_dir = "/mnt/euler/SUGAR/SuGaR/output/track_241213T8/"
# out_dir = "/mnt/euler/SUGAR/SuGaR/output/track_220920T1/"
# out_dir_2 = "/media/dalco/data/SUGAR/SuGaR/output/track_240906T3_colorEdit1/"

cmr_list = [18]


def merge_seq(bg_label="_w", tracking_folder="tracking_999faces_0to200"):

    for cmr_idx in cmr_list:
        merge_dir = out_dir + f"render/{cmr_idx}/"
        os.makedirs(merge_dir, exist_ok=True)

        if os.path.exists(merge_dir + "pad.txt"):
            pad = np.loadtxt(merge_dir + "pad.txt").astype(np.int64)
        else:
            pad = np.array(pix_pad)
            np.savetxt(merge_dir + "pad.txt", pad)

        for frame in tqdm(range(frame_0, frame_end, interval)):
            gt_img = cv2.imread(in_dir + f"{frame:04d}/images/img_{cmr_idx:04d}.jpg")
            # gt_img = cv2.imread(in_dir + f"0166/images/img_0013.jpg")
            gt_img = gt_img[pad[0]:-pad[1], pad[2]:-pad[3]]

            # if 0 and os.path.exists(out_dir + f"{frame:04d}_update/"):
            #     render_img = cv2.imread(out_dir + f"{frame:04d}_update/render/render_{cmr_idx:06d}.jpg")
            # else:
            render_img = cv2.imread(out_dir + f"{frame:04d}/render{bg_label}/render_{cmr_idx:06d}.jpg")
            # render_img = cv2.imread(out_dir + f"render_rotating/render_0166_{frame}.jpg")
            render_img = render_img[pad[0]:-pad[1], pad[2]:-pad[3]]
            # # render_img_2 = cv2.imread(out_dir_2 + f"{frame:04d}/render{bg_label}/render_{cmr_idx:06d}.jpg")
            # render_img_2 = cv2.imread(out_dir + f"render_rotating/0166_{frame}.jpg")
            # render_img_2 = render_img_2[pad[0]:-pad[1], pad[2]:-pad[3]]

            normal_img = cv2.imread(out_dir + f"render/{cmr_idx}_{frame:04d}.jpg")
            normal_img = normal_img[pad[0]:-pad[1], pad[2]:-pad[3]]

            track_img = cv2.imread(out_dir + f"tracking/{tracking_folder}/{cmr_idx}_{frame:04d}.jpg")
            track_img = track_img[pad[0]:-pad[1], pad[2]:-pad[3]]

            merge_img = np.hstack((gt_img, render_img, normal_img, track_img))
            # merge_img = np.hstack((gt_img, render_img, normal_img))
            # merge_img = np.hstack((render_img, normal_img))
            # merge_img = np.hstack((gt_img, render_img))
            # merge_img = np.hstack((gt_img, render_img, render_img_2))

            if resize:
                merge_size = np.asarray(merge_img.shape[:2]) // 2
                if merge_size[0] % 2 == 1:
                    merge_size[0] = merge_size[0] + 1
                if merge_size[1] % 2 == 1:
                    merge_size[1] = merge_size[1] + 1
                merge_img = cv2.resize(merge_img, merge_size[::-1])
                # merge_img = cv2.resize(merge_img, (2932, 1296))

            cv2.imwrite(merge_dir + f"{(frame // interval):04d}.jpg", merge_img)


def merge_comp_render():
    for cmr_i in cmr_list:
        merge_dir = f"/media/dalco/data/SUGAR/video/comp/1028T4/{cmr_i}/"
        os.makedirs(merge_dir, exist_ok=True)
        if os.path.exists(merge_dir + "pad.txt"):
            pad = np.loadtxt(merge_dir + "pad.txt").astype(np.int64)
        else:
            pad = np.array(pix_pad)
            np.savetxt(merge_dir + "pad.txt", pad)

        gt_dir = "/mnt/euler/SUGAR/data/mocap/track_241028_Take4/"
        hrf_dir = "/media/dalco/Data_Chengwei/humanrf/out/mocap_241028_Take4/"
        dygs_dir = "/media/dalco/Data_Chengwei_21/Dy3DGS/mocap_241028_Take4/"
        pa_dir = "/media/dalco/Data_Chengwei_21/PhysAvatar_SMPL/mocap_241028_Take4/"
        tdgs_dir = "/media/dalco/data/2DGS/2d-gaussian-splatting/output/1028T4_seq/"
        our_dir = "/mnt/euler/SUGAR/SuGaR/output/track_241028T4/"

        for f_idx in tqdm(range(frame_0, frame_end, interval)):

            gt_img = cv2.imread(gt_dir + f"{f_idx:04d}/images/img_{cmr_i:04d}.jpg")
            gt_img = gt_img[pad[0]:-pad[1], pad[2]:-pad[3]]

            hrf_img = cv2.imread(hrf_dir + f"results/test_frames/Cam{cmr_i:03d}_rgb{(f_idx - frame_0):06d}.jpg")
            hrf_img = hrf_img[pad[0]:-pad[1], pad[2]:-pad[3]]

            dygs_img = cv2.imread(dygs_dir + f"{cmr_i}/{f_idx:06d}.jpg")
            dygs_img = dygs_img[pad[0]:-pad[1], pad[2]:-pad[3]]

            pa_img = cv2.imread(pa_dir + f"{cmr_i}/{f_idx:06d}.jpg")
            pa_img = pa_img[pad[0]:-pad[1], pad[2]:-pad[3]]

            # tdgs_img = cv2.imread(tdgs_dir + f"{f_idx:04d}/train_w/ours_10000/renders_w/{cmr_i:05d}.jpg")
            tdgs_img = cv2.imread(tdgs_dir + f"{f_idx:04d}/train/ours_8000/renders_w/{cmr_i:05d}.jpg")
            tdgs_img = tdgs_img[pad[0]:-pad[1], pad[2]:-pad[3]]

            our_img = cv2.imread(our_dir + f"{f_idx:04d}/render_w/render_{cmr_i:06d}.jpg")
            our_img = our_img[pad[0]:-pad[1], pad[2]:-pad[3]]

            r1_img = np.hstack((gt_img, hrf_img, tdgs_img))
            r2_img = np.hstack((dygs_img, pa_img, our_img))
            merge_img = np.vstack((r1_img, r2_img))

            if resize:
                merge_size = np.asarray(merge_img.shape[:2]) // 2
                if merge_size[0] % 2 == 1:
                    merge_size[0] = merge_size[0] + 1
                if merge_size[1] % 2 == 1:
                    merge_size[1] = merge_size[1] + 1
                merge_img = cv2.resize(merge_img, merge_size[::-1])

            cv2.imwrite(merge_dir + f"{(f_idx // interval):04d}.jpg", merge_img)


def merge_comp_geo():
    merge_dir = f"/media/dalco/data/SUGAR/video/comp/0906T3/mesh/"
    os.makedirs(merge_dir, exist_ok=True)
    if os.path.exists(merge_dir + "pad.txt"):
        pad = np.loadtxt(merge_dir + "pad.txt").astype(np.int64)
    else:
        pad = np.array(pix_pad)
        np.savetxt(merge_dir + "pad.txt", pad)

    gt_dir = merge_dir + "gt/render/"
    hrf_dir = "/media/dalco/Data_Chengwei/humanrf/out/mocap_240906_Take3/results/mesh/simp/render/"
    dygs_dir = "/media/dalco/Data_Chengwei_21/Dy3DGS/mocap_240906_Take3/meshes/render/"
    pa_dir = "/media/dalco/Data_Chengwei_21/PhysAvatar_SMPL/mocap_240906_Take3_w/meshes_exp_gstar_with_lbs/render/"
    tdgs_dir = merge_dir + "2dgs/render/"
    # our_dir = "/mnt/euler/SUGAR/SuGaR/output/track_241028T4/"
    our_dir = "//output/track_240906T3_update_re/"

    for f_idx in tqdm(range(frame_0, frame_end, interval)):

        gt_img = cv2.imread(gt_dir + f"mesh-f{f_idx:05d}.jpg")
        gt_img = gt_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        # hrf_img = cv2.imread(hrf_dir + f"{(f_idx-frame_0):06d}_sm2.jpg")
        hrf_img = cv2.imread(hrf_dir + f"mesh_{(f_idx-20):06d}_smooth_40k.jpg")
        hrf_img = hrf_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        dygs_img = cv2.imread(dygs_dir + f"{f_idx:06d}.jpg")
        dygs_img = dygs_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        pa_img = cv2.imread(pa_dir + f"{f_idx:06d}.jpg")
        pa_img = pa_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        tdgs_img = cv2.imread(tdgs_dir + f"fuse_100k_{f_idx:04d}.jpg")
        tdgs_img = tdgs_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        our_img = cv2.imread(our_dir + f"render/30_{f_idx:04d}.jpg")
        our_img = our_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        r1_img = np.hstack((gt_img, hrf_img, tdgs_img))
        r2_img = np.hstack((dygs_img, pa_img, our_img))
        merge_img = np.vstack((r1_img, r2_img))

        if resize:
            merge_size = np.asarray(merge_img.shape[:2]) // 2
            if merge_size[0] % 2 == 1:
                merge_size[0] = merge_size[0] + 1
            if merge_size[1] % 2 == 1:
                merge_size[1] = merge_size[1] + 1
            merge_img = cv2.resize(merge_img, merge_size[::-1])

        cv2.imwrite(merge_dir + f"{(f_idx // interval):04d}.jpg", merge_img)


def merge_ablation_geo():
    merge_dir = f"/media/dalco/data/SUGAR/video/ablation/0724T12/mesh_1r/"
    os.makedirs(merge_dir, exist_ok=True)
    if os.path.exists(merge_dir + "pad.txt"):
        pad = np.loadtxt(merge_dir + "pad.txt").astype(np.int64)
    else:
        pad = np.array(pix_pad)
        np.savetxt(merge_dir + "pad.txt", pad)

    cmr_i = 16

    gt_dir = "/media/dalco/data/SUGAR/video/ablation/0724T12/mesh/gt/"
    woU_dir = "/mnt/server02/GSTAR/track_240724T12_wo_update/"
    woR_dir = "/mnt/server02/GSTAR/track_240724T12_wo_remesh_re/"
    woF_dir = "/mnt/server02/GSTAR/track_240724T12_wo_flow/"
    our_dir = "//output/track_240724T12_update_floor/"

    for f_idx in tqdm(range(frame_0, frame_end, interval)):

        gt_img = cv2.imread(gt_dir + f"render/mesh-f{f_idx:05d}.jpg")
        gt_img = gt_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        woU_img = cv2.imread(woU_dir + f"render/{cmr_i}_{f_idx:04d}.jpg")
        woU_img = woU_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        woR_img = cv2.imread(woR_dir + f"render/{cmr_i}_{f_idx:04d}.jpg")
        woR_img = woR_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        woF_img = cv2.imread(woF_dir + f"render/{cmr_i}_{f_idx:04d}.jpg")
        woF_img = woF_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        our_img = cv2.imread(our_dir + f"render/{cmr_i}_{f_idx:04d}.jpg")
        our_img = our_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        # r1_img = np.hstack((gt_img, woU_img, woR_img))
        # r2_img = np.hstack((woF_img, woF_img, our_img))
        # merge_img = np.vstack((r1_img, r2_img))
        merge_img = np.hstack((gt_img, woU_img, woR_img, woF_img, our_img))

        if resize:
            merge_size = np.asarray(merge_img.shape[:2]) // 2
            if merge_size[0] % 2 == 1:
                merge_size[0] = merge_size[0] + 1
            if merge_size[1] % 2 == 1:
                merge_size[1] = merge_size[1] + 1
            merge_img = cv2.resize(merge_img, merge_size[::-1])

        cv2.imwrite(merge_dir + f"{(f_idx // interval):04d}.jpg", merge_img)


def merge_ablation_render():
    merge_dir = f"/media/dalco/data/SUGAR/video/ablation/0724T10/render_1r/"
    os.makedirs(merge_dir, exist_ok=True)
    if os.path.exists(merge_dir + "pad.txt"):
        pad = np.loadtxt(merge_dir + "pad.txt").astype(np.int64)
    else:
        pad = np.array(pix_pad)
        np.savetxt(merge_dir + "pad.txt", pad)

    cmr_i = 18

    gt_dir = "/media/dalco/Data_Chengwei_21/SuGaR/data/track_240724_Take10/"
    woU_dir = "/mnt/server02/GSTAR/track_240724T10_wo_update/"
    woR_dir = "/mnt/server02/GSTAR/track_240724T10_wo_remesh_re/"
    woF_dir = "/mnt/server02/GSTAR/track_240724T10_wo_flow/"
    our_dir = "//output/track_240724T10_update/"

    for f_idx in tqdm(range(frame_0, frame_end, interval)):

        gt_img = cv2.imread(gt_dir + f"{f_idx:04d}/images/img_{cmr_i:04d}.jpg")
        gt_img = gt_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        woU_img = cv2.imread(woU_dir + f"{f_idx:04d}/render_w/render_{cmr_i:06d}.jpg")
        woU_img = woU_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        woR_img = cv2.imread(woR_dir + f"{f_idx:04d}/render_w/render_{cmr_i:06d}.jpg")
        woR_img = woR_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        woF_img = cv2.imread(woF_dir + f"{f_idx:04d}/render_w/render_{cmr_i:06d}.jpg")
        woF_img = woF_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        our_img = cv2.imread(our_dir + f"{f_idx:04d}/render_w/render_{cmr_i:06d}.jpg")
        our_img = our_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        # r1_img = np.hstack((gt_img, woU_img, woR_img))
        # r2_img = np.hstack((woF_img, woF_img, our_img))
        # merge_img = np.vstack((r1_img, r2_img))
        merge_img = np.hstack((gt_img, woU_img, woR_img, woF_img, our_img))

        if resize:
            merge_size = np.asarray(merge_img.shape[:2]) // 2
            if merge_size[0] % 2 == 1:
                merge_size[0] = merge_size[0] + 1
            if merge_size[1] % 2 == 1:
                merge_size[1] = merge_size[1] + 1
            merge_img = cv2.resize(merge_img, merge_size[::-1])

        cv2.imwrite(merge_dir + f"{(f_idx // interval):04d}.jpg", merge_img)


def merge_comp_tracking():
    merge_dir = f"/media/dalco/data/SUGAR/video/comp/1016T4/"
    os.makedirs(merge_dir, exist_ok=True)
    if os.path.exists(merge_dir + "pad.txt"):
        pad = np.loadtxt(merge_dir + "pad.txt").astype(np.int64)
    else:
        pad = np.array(pix_pad)
        np.savetxt(merge_dir + "pad.txt", pad)

    cmr_i = 18
    gt_dir = "/mnt/euler/SUGAR/data/mocap/track_241016_Take4/"
    tracking_dir = "/mnt/euler/SUGAR/SuGaR/output/track_241016T4_SHreg/tracking/"
    dygs_dir = tracking_dir + "dy3dgs_pred_points3d/"
    pa_dir = tracking_dir + "exp_gstar_with_lbs_pred_points3d/"
    our_dir = tracking_dir + "tracking_kp_id_bary/"

    for f_idx in tqdm(range(frame_0, frame_end, interval)):

        gt_img = cv2.imread(gt_dir + f"{f_idx:04d}/images/img_{cmr_i:04d}.jpg")
        gt_img = gt_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        dygs_img = cv2.imread(dygs_dir + f"{cmr_i}_{f_idx:04d}.jpg")
        dygs_img = dygs_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        pa_img = cv2.imread(pa_dir + f"{cmr_i}_{f_idx:04d}.jpg")
        pa_img = pa_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        our_img = cv2.imread(our_dir + f"{cmr_i}_{f_idx:04d}.jpg")
        our_img = our_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        merge_img = np.hstack((gt_img, dygs_img, pa_img, our_img))

        if resize:
            merge_size = np.asarray(merge_img.shape[:2]) // 2
            if merge_size[0] % 2 == 1:
                merge_size[0] = merge_size[0] + 1
            if merge_size[1] % 2 == 1:
                merge_size[1] = merge_size[1] + 1
            merge_img = cv2.resize(merge_img, merge_size[::-1])

        cv2.imwrite(merge_dir + f"{(f_idx // interval):04d}.jpg", merge_img)


def merge_rot(frame_0=220):
    merge_dir = out_dir + f"render_rotating/{frame_0}/"
    os.makedirs(merge_dir, exist_ok=True)

    if os.path.exists(merge_dir + "pad.txt"):
        pad = np.loadtxt(merge_dir + "pad.txt").astype(np.int64)
    else:
        pad = np.array(pix_pad)
        np.savetxt(merge_dir + "pad.txt", pad)

    gt_img = cv2.imread(in_dir + f"{frame_0:04d}/images/img_{cmr_list[0]:04d}.jpg")
    gt_img = gt_img[pad[0]:-pad[1], pad[2]:-pad[3]]

    for cmr_idx in tqdm(range(120)):
        render_img = cv2.imread(out_dir + f"render_rotating/render_{frame_0:04d}_{cmr_idx}.jpg")
        render_img = render_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        normal_img = cv2.imread(out_dir + f"render_rotating/{frame_0:04d}_{cmr_idx}.jpg")
        normal_img = normal_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        track_img = cv2.imread(out_dir + f"tracking_rotating/{frame_0:04d}_{cmr_idx}.jpg")
        track_img = track_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        merge_img = np.hstack((gt_img, render_img, normal_img, track_img))
        if resize:
            merge_size = np.asarray(merge_img.shape[:2]) // 2
            if merge_size[0] % 2 == 1:
                merge_size[0] = merge_size[0] + 1
            if merge_size[1] % 2 == 1:
                merge_size[1] = merge_size[1] + 1
            merge_img = cv2.resize(merge_img, merge_size[::-1])
            # merge_img = cv2.resize(merge_img, (3908, 1296))
        cv2.imwrite(merge_dir + f"{frame_0:04d}_{cmr_idx}.jpg", merge_img)


def merge_vestir():
    merge_dir = f"//output/vestir_G11M4_Take3/comp/"
    os.makedirs(merge_dir, exist_ok=True)
    if os.path.exists(merge_dir + "pad.txt"):
        pad = np.loadtxt(merge_dir + "pad.txt").astype(np.int64)
    else:
        pad = np.array(pix_pad)
        np.savetxt(merge_dir + "pad.txt", pad)

    cmr_i = 15
    cmr_name = 5

    gt_dir = "/media/dalco/Data_Chengwei_21/SuGaR/data/vestir_G11M4_Take3/"
    ms_dir = "/media/dalco/Data_Chengwei/humanrf/in/vestir_G11M4_Take3/frames/"
    our_dir = "//output/vestir_G11M4_Take3/"

    for f_idx in tqdm(range(frame_0, frame_end, interval)):

        gt_img = cv2.imread(gt_dir + f"{f_idx:04d}/images/img_{cmr_i:04d}.jpg")
        gt_img = gt_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        ms_img = cv2.imread(ms_dir + f"render_rgb/{(f_idx+1):d}_{cmr_name}.jpg")
        ms_img = ms_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        ms_geo = cv2.imread(ms_dir + f"render_norm/mesh-f{(f_idx+1):05d}_{cmr_name}.jpg")
        ms_geo = ms_geo[pad[0]:-pad[1], pad[2]:-pad[3]]

        our_img = cv2.imread(our_dir + f"{f_idx:04d}/render_w/render_{cmr_i:06d}.jpg")
        our_img = our_img[pad[0]:-pad[1], pad[2]:-pad[3]]

        our_geo = cv2.imread(our_dir + f"render/{cmr_i:d}_{f_idx:04d}.jpg")
        our_geo = our_geo[pad[0]:-pad[1], pad[2]:-pad[3]]

        merge_img = np.hstack((gt_img, ms_geo, ms_img, our_geo, our_img))

        if resize:
            merge_size = np.asarray(merge_img.shape[:2]) // 2
            if merge_size[0] % 2 == 1:
                merge_size[0] = merge_size[0] + 1
            if merge_size[1] % 2 == 1:
                merge_size[1] = merge_size[1] + 1
            merge_img = cv2.resize(merge_img, merge_size[::-1])

        cv2.imwrite(merge_dir + f"{(f_idx // interval):04d}.jpg", merge_img)


merge_vestir()
# merge_seq()
# merge_rot()

# merge_comp_render()
# merge_comp_geo()
# merge_comp_tracking()

# merge_ablation_geo()
# merge_ablation_render()

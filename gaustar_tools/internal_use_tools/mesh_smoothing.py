import numpy as np
import open3d
import trimesh
from tqdm import tqdm


def smoothing():
    for i in range(1):
        # mesh = open3d.io.read_triangle_mesh(f"/media/dalco/Data_Chengwei/humanrf/out/mocap_240906_Take8/results/fusion/extract_0089_re6.obj")
        mesh = open3d.io.read_triangle_mesh("/media/dalco/data/SUGAR/fig/ablation/0724T12/mesh/mesh-f00235.ply")
        # mesh = open3d.io.read_triangle_mesh("/media/dalco/Data_Chengwei/humanrf/out/mocap_240906_Take8/results/fusion/extract_0089_re.obj")

        mesh = mesh.simplify_quadric_decimation(100_000)
        # mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=7)

        # open3d.io.write_triangle_mesh(f"/media/dalco/Data_Chengwei/humanrf/out/mocap_240906_Take8/results/fusion/extract_0089_re6_0-7.obj", mesh)
        open3d.io.write_triangle_mesh("/media/dalco/data/SUGAR/video/comp/0906T3/mesh/mesh-f00235_simp_sm.ply", mesh)
        # open3d.io.write_triangle_mesh("/media/dalco/Data_Chengwei/humanrf/out/mocap_240906_Take8/results/fusion/extract_0089_re_sm.obj", mesh)


def smoothing_gt():
    for i in tqdm(range(1, 229, 2)):
        mesh = open3d.io.read_triangle_mesh(f"/mnt/server02/GSTAR/data_copy_juan/smoothmeshes_1028_T4/mesh-f{(i+1):05d}.ply")

        mesh = mesh.simplify_quadric_decimation(100_000)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)  # or 7

        open3d.io.write_triangle_mesh(f"/media/dalco/data/SUGAR/video/comp/1028T4/mesh/gt/mesh-f{i:05d}.ply", mesh)


def smoothing_hrf():
    for i in tqdm(range(110, 290-20)):
        mesh = open3d.io.read_triangle_mesh(f"/media/dalco/Data_Chengwei/humanrf/out/mocap_240906_Take3/results/mesh/simp/mesh_{i:06d}_smooth_40k.obj")

        # mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=2)

        open3d.io.write_triangle_mesh(f"/media/dalco/Data_Chengwei/humanrf/out/mocap_240906_Take3/results/mesh/smooth/{i:06d}_sm2.obj", mesh)


def editing():
    mesh = trimesh.load_mesh("/media/dalco/data/SUGAR/fig/teaser/ellipsoid/ellipsoid.obj")
    mesh.vertices[:, 0] *= 0.5
    mesh.vertices[:, 2] *= 0.2
    vertex_colors = np.zeros_like(mesh.vertices)
    vertex_colors[:, 0] = 0
    vertex_colors[:, 1] = 112 / 255
    vertex_colors[:, 2] = 192 / 255
    new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors)
    new_mesh.export("/media/dalco/data/SUGAR/fig/teaser/ellipsoid/ellipsoid_02z05x.obj")


smoothing_gt()
# smoothing_hrf()

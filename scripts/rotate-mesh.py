import numpy as np
import open3d as o3d

trans = np.load("/home/daizhirui/Data/NewerCollege-DLIO/rotated/T_rotated_to_original.npy")
mesh = o3d.io.read_triangle_mesh(
    "/home/daizhirui/D/GoogleDrive/Documents/UCSD/Research/ERL/SDF/Neural-SDF/reconstructed_mesh_result/newer_college-pin_slam.ply"
)
mesh.transform(np.linalg.inv(trans))

mesh_ref = o3d.io.read_triangle_mesh("/home/daizhirui/D/GoogleDrive/Documents/UCSD/Research/ERL/SDF/Neural-SDF/reconstructed_mesh_result/newer_college-our.ply")
mesh_ref.paint_uniform_color([1, 0, 0])

o3d.io.write_triangle_mesh(
    "/home/daizhirui/D/GoogleDrive/Documents/UCSD/Research/ERL/SDF/Neural-SDF/reconstructed_mesh_result/newer_college-pin_slam-rotated.ply",
    mesh,
)

o3d.visualization.draw_geometries([mesh, mesh_ref])

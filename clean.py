import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("outputs/point_cloud.ply")
normals = np.load("outputs/normals.npy")

min_len = min(len(pcd.points), len(normals))
pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[:min_len])
normals = normals[:min_len]
pcd.normals = o3d.utility.Vector3dVector(normals)

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)

clean_pcd = pcd.select_by_index(ind)
clean_normals = normals[ind]

o3d.io.write_point_cloud("outputs/clean_room.ply", clean_pcd)
np.save("outputs/clean_normals.npy", clean_normals)

normal_colors = (clean_normals + 1.0) / 2.0
clean_pcd.colors = o3d.utility.Vector3dVector(normal_colors)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Cleaned Point Cloud")
vis.add_geometry(clean_pcd)
vis.get_render_option().background_color = np.array([0.05, 0.05, 0.05])
vis.run()
vis.destroy_window()
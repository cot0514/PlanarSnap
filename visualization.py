import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("outputs/clean_room.ply")

normals = np.load("outputs/clean_normals.npy")

if len(pcd.points) != len(normals):
    print(f"({len(pcd.points)}), ({len(normals)}) is not correct")
    min_len = min(len(pcd.points), len(normals))
    points = np.asarray(pcd.points)[:min_len]
    normals = normals[:min_len]
    pcd.points = o3d.utility.Vector3dVector(points)
else:
    print(f"{len(normals)} normal vector loaded successfully.")

normal_colors = (normals + 1.0) / 2.0
pcd.colors = o3d.utility.Vector3dVector(normal_colors)
pcd.normals = o3d.utility.Vector3dVector(normals)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="BBSplat Normal Inspector")
vis.add_geometry(pcd)

opt = vis.get_render_option()
opt.background_color = np.array([0.05, 0.05, 0.05])
opt.point_size = 3.0 
opt.light_on = True

vis.run()
vis.destroy_window()
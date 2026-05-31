import numpy as np
import torch
from plyfile import PlyData


def quaternion_to_matrix(quaternions):
    r, i, j, k = quaternions.unbind(-1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o3 = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o3.reshape(quaternions.shape[:-1] + (3, 3))


ply_path = "output/room_output/point_cloud/iteration_32000/point_cloud.ply"

plydata = PlyData.read(ply_path)

rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
quats = np.vstack([plydata.elements[0].data[name] for name in rot_names]).T
quats = torch.tensor(quats, dtype=torch.float32)

quats = torch.nn.functional.normalize(quats, p=2, dim=-1)

R = quaternion_to_matrix(quats)
normals = R[:, :, 2]

save_path = "output/room_output/point_cloud/iteration_32000/normals.npy"
np.save(save_path, normals.numpy())

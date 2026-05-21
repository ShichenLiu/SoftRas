import numpy as np
import torch
import torch.nn.functional as F


def look_at(vertices, eye, at=[0, 0, 0], up=[0, 1, 0]):
    """
    "Look at" transformation of vertices.
    """
    if vertices.ndimension() != 3:
        raise ValueError('vertices Tensor should have 3 dimensions')

    device = vertices.device
    dtype = vertices.dtype

    def _to_tensor(value):
        if isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=dtype, device=device)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=device, dtype=dtype)
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        raise TypeError('Expected a list, tuple, numpy.ndarray, or torch.Tensor')

    at = _to_tensor(at)
    up = _to_tensor(up)
    eye = _to_tensor(eye)

    batch_size = vertices.shape[0]
    if eye.ndimension() == 1:
        eye = eye[None, :].repeat(batch_size, 1)
    if at.ndimension() == 1:
        at = at[None, :].repeat(batch_size, 1)
    if up.ndimension() == 1:
        up = up[None, :].repeat(batch_size, 1)

    # create new axes
    # eps is chosen as 1e-5 to match the chainer version
    z_axis = F.normalize(at - eye, eps=1e-5, dim=1)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5, dim=1)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5, dim=1)

    # create rotation matrix: [bs, 3, 3]
    r = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye[:, None, :]
    vertices = vertices - eye
    vertices = torch.matmul(vertices, r.transpose(1, 2))

    return vertices

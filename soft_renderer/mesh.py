import numpy as np
import torch
import torch.nn.functional as F

import soft_renderer.functional as srf


def _default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Mesh(object):
    '''
    A simple class for creating and manipulating trimesh objects
    '''
    def __init__(self, vertices, faces, textures=None, texture_res=1, texture_type='surface'):
        '''
        vertices, faces and textures(if not None) are expected to be Tensor objects
        '''
        self._vertices = vertices
        self._faces = faces

        if isinstance(self._vertices, np.ndarray):
            self._vertices = torch.from_numpy(self._vertices).float().to(_default_device())
        if isinstance(self._faces, np.ndarray):
            self._faces = torch.from_numpy(self._faces.astype(np.int32)).to(_default_device())

        if self._vertices.ndimension() == 2:
            self._vertices = self._vertices[None, :, :]
        if self._faces.ndimension() == 2:
            self._faces = self._faces[None, :, :]

        self.device = self._vertices.device
        self._vertices = self._vertices.to(self.device)
        self._faces = self._faces.to(device=self.device, dtype=torch.int32).contiguous()
        self.texture_type = texture_type

        self.batch_size = self._vertices.shape[0]
        self.num_vertices = self._vertices.shape[1]
        self.num_faces = self._faces.shape[1]

        # create textures
        if textures is None:
            if texture_type == 'surface':
                self._textures = torch.ones(
                    self.batch_size,
                    self.num_faces,
                    texture_res ** 2,
                    3,
                    dtype=self._vertices.dtype,
                    device=self.device,
                )
                self.texture_res = texture_res
            elif texture_type == 'vertex':
                self._textures = torch.ones(
                    self.batch_size,
                    self.num_vertices,
                    3,
                    dtype=self._vertices.dtype,
                    device=self.device,
                )
                self.texture_res = 1  # vertex color doesn't need resolution
            else:
                raise ValueError('texture type not applicable')
        else:
            if isinstance(textures, np.ndarray):
                textures = torch.from_numpy(textures).float()
            if textures.ndimension() == 3 and texture_type == 'surface':
                textures = textures[None, :, :, :]
            if textures.ndimension() == 2 and texture_type == 'vertex':
                textures = textures[None, :, :]
            self._textures = textures.to(self.device).contiguous()
            if texture_type == 'surface':
                self.texture_res = int(np.sqrt(self._textures.shape[2]))
            elif texture_type == 'vertex':
                self.texture_res = 1
            else:
                raise ValueError('texture type not applicable')

    @classmethod
    def from_obj(cls, filename_obj, normalization=False, load_texture=False, texture_res=1, texture_type='surface'):
        '''
        Create a Mesh object from a .obj file
        '''
        if load_texture:
            vertices, faces, textures = srf.load_obj(
                filename_obj,
                normalization=normalization,
                texture_res=texture_res,
                load_texture=True,
                texture_type=texture_type,
            )
        else:
            vertices, faces = srf.load_obj(
                filename_obj,
                normalization=normalization,
                texture_res=texture_res,
                load_texture=False,
            )
            textures = None
        return cls(vertices, faces, textures, texture_res, texture_type)

    def save_obj(self, filename_obj, save_texture=False, texture_res_out=16):
        if self.batch_size != 1:
            raise ValueError('Could not save when batch size > 1')
        if save_texture:
            srf.save_obj(
                filename_obj,
                self.vertices[0],
                self.faces[0],
                textures=self.textures[0],
                texture_res=texture_res_out,
                texture_type=self.texture_type,
            )
        else:
            srf.save_obj(filename_obj, self.vertices[0], self.faces[0], textures=None)

    @property
    def faces(self):
        return self._faces

    @property
    def vertices(self):
        return self._vertices

    @property
    def textures(self):
        return self._textures

    @property
    def face_vertices(self):
        return srf.face_vertices(self.vertices, self.faces)

    @property
    def surface_normals(self):
        v10 = self.face_vertices[:, :, 0] - self.face_vertices[:, :, 1]
        v12 = self.face_vertices[:, :, 2] - self.face_vertices[:, :, 1]
        return F.normalize(torch.cross(v12, v10, dim=2), p=2, dim=2, eps=1e-6)

    @property
    def vertex_normals(self):
        return srf.vertex_normals(self.vertices, self.faces)

    @property
    def face_textures(self):
        if self.texture_type in ['surface']:
            return self.textures
        elif self.texture_type in ['vertex']:
            return srf.face_vertices(self.textures, self.faces)
        else:
            raise ValueError('texture type not applicable')

    def voxelize(self, voxel_size=32):
        face_vertices_norm = self.face_vertices * voxel_size / (voxel_size - 1) + 0.5
        return srf.voxelization(face_vertices_norm, voxel_size, False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import soft_renderer.functional as srf


class AmbientLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1,1,1)):
        super(AmbientLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color

    def forward(self, light):
        return srf.ambient_lighting(light, self.light_intensity, self.light_color)


class DirectionalLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1,1,1), light_direction=(0,1,0)):
        super(DirectionalLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color
        self.light_direction = light_direction

    def forward(self, light, normals):
        return srf.directional_lighting(light, normals,
                                        self.light_intensity, self.light_color, 
                                        self.light_direction)


class Lighting(nn.Module):
    def __init__(self, light_mode='surface',
                 intensity_ambient=0.5, color_ambient=[1,1,1],
                 intensity_directionals=0.5, color_directionals=[1,1,1],
                 directions=[0,1,0]):
        super(Lighting, self).__init__()

        if light_mode not in ['surface', 'vertex']:
            raise ValueError('Lighting mode only support surface and vertex')

        self.light_mode = light_mode
        self.ambient = AmbientLighting(intensity_ambient, color_ambient)
        self.directionals = nn.ModuleList([DirectionalLighting(intensity_directionals,
                                                               color_directionals,
                                                               directions)])

    def forward(self, mesh):
        if self.light_mode == 'surface':
            light = torch.zeros_like(mesh.faces, dtype=torch.float32).to(mesh.device)
            light = light.contiguous()
            light = self.ambient(light)
            for directional in self.directionals:
                light = directional(light, mesh.surface_normals)
            mesh.textures = mesh.textures * light[:, :, None, :]

        elif self.light_mode == 'vertex':
            light = torch.zeros_like(mesh.vertices, dtype=torch.float32).to(mesh.device)
            light = light.contiguous()
            light = self.ambient(light)
            for directional in self.directionals:
                light = directional(light, mesh.vertex_normals)
            mesh.textures = mesh.textures * light

        return mesh
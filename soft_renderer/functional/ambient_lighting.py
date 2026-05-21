import numpy as np
import torch


def ambient_lighting(light, light_intensity=0.5, light_color=(1, 1, 1)):
    device = light.device
    dtype = light.dtype

    if isinstance(light_color, (tuple, list)):
        light_color = torch.tensor(light_color, dtype=dtype, device=device)
    elif isinstance(light_color, np.ndarray):
        light_color = torch.from_numpy(light_color).to(device=device, dtype=dtype)
    else:
        light_color = light_color.to(device=device, dtype=dtype)

    if light_color.ndimension() == 1:
        light_color = light_color[None, :]

    light = light + light_intensity * light_color[:, None, :]
    return light  # [nb, :, 3]

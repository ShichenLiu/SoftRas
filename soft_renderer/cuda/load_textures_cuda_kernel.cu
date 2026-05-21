#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAException.h>

#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {
template <typename scalar_t>
__global__ void load_textures_cuda_kernel(
    const scalar_t* __restrict__ image,
    const scalar_t* __restrict__ faces,
    const int32_t* __restrict__ is_update,
    scalar_t* __restrict__ textures,
    size_t texture_size,
    size_t texture_res,
    size_t image_height,
    size_t image_width) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 3 >= texture_size) {
      return;
  }
  const int R = texture_res;
  const int fn = i / (R * R);
  const int w_y = (i % (R * R)) / R;
  const int w_x = i % R;
  // compute barycentric coordinate
  scalar_t w0, w1, w2;
  if (w_x + w_y < R) {
      w0 = (w_x + scalar_t(1.0 / 3.0)) / R;
      w1 = (w_y + scalar_t(1.0 / 3.0)) / R;
      w2 = 1. - w0 - w1;
  } else {
      w0 = ((R - 1. - w_x) + scalar_t(2.0 / 3.0)) / R;
      w1 = ((R - 1. - w_y) + scalar_t(2.0 / 3.0)) / R;
      w2 = 1. - w0 - w1;
  }
  const scalar_t* face = &faces[fn * 3 * 2];
  scalar_t* texture = &textures[i * 3];
  if (is_update[fn] == 0) {
      return;
  }

  const scalar_t pos_x = (
      (face[2 * 0 + 0] * w0 + face[2 * 1 + 0] * w1 + face[2 * 2 + 0] * w2) * (image_width - 1));
  const scalar_t pos_y = (
      (face[2 * 0 + 1] * w0 + face[2 * 1 + 1] * w1 + face[2 * 2 + 1] * w2) * (image_height - 1));
  if (1) {
      /* bilinear sampling */
      const scalar_t weight_x1 = pos_x - static_cast<int>(pos_x);
      const scalar_t weight_x0 = 1 - weight_x1;
      const scalar_t weight_y1 = pos_y - static_cast<int>(pos_y);
      const scalar_t weight_y0 = 1 - weight_y1;
      for (int k = 0; k < 3; k++) {
          scalar_t c = 0;
          c += image[((static_cast<int>(pos_y) * image_width) + static_cast<int>(pos_x)) * 3 + k] * (weight_x0 * weight_y0);
          c += image[((static_cast<int>(pos_y + 1) * image_width) + static_cast<int>(pos_x)) * 3 + k] * (weight_x0 * weight_y1);
          c += image[((static_cast<int>(pos_y) * image_width) + static_cast<int>(pos_x) + 1) * 3 + k] * (weight_x1 * weight_y0);
          c += image[((static_cast<int>(pos_y + 1) * image_width) + static_cast<int>(pos_x) + 1) * 3 + k] * (weight_x1 * weight_y1);
          texture[k] = c;
      }
  } else {
      /* nearest neighbor */
      const int pos_xi = round(pos_x);
      const int pos_yi = round(pos_y);
      for (int k = 0; k < 3; k++) {
          texture[k] = image[(pos_yi * image_width + pos_xi) * 3 + k];
      }
  }
}
} // namespace

at::Tensor load_textures_cuda(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor is_update) {
    // texture_size = size of the textures tensor
    const auto texture_size = textures.numel();
    const auto texture_res = static_cast<int>(std::sqrt(static_cast<double>(textures.size(1))));
    const auto image_height = image.size(0);
    const auto image_width = image.size(1);

    const int threads = 1024;
    const dim3 blocks((texture_size / 3 - 1) / threads + 1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "load_textures_cuda", ([&] {
        load_textures_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            image.data_ptr<scalar_t>(),
            faces.data_ptr<scalar_t>(),
            is_update.data_ptr<int32_t>(),
            textures.data_ptr<scalar_t>(),
            texture_size,
            texture_res,
            image_height,
            image_width);
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return textures;
}

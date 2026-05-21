#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

// CUDA forward declarations

at::Tensor create_texture_image_cuda(
        at::Tensor vertices_all,
        at::Tensor textures,
        at::Tensor image,
        float eps);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOATING(x) TORCH_CHECK(at::isFloatingType((x).scalar_type()), #x " must be a floating point tensor")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_SAME_DEVICE(x, y) TORCH_CHECK((x).device() == (y).device(), #x " and " #y " must be on the same device")
#define CHECK_SAME_DTYPE(x, y) TORCH_CHECK((x).scalar_type() == (y).scalar_type(), #x " and " #y " must have the same dtype")

at::Tensor create_texture_image(
        at::Tensor vertices_all,
        at::Tensor textures,
        at::Tensor image,
        float eps) {

    CHECK_INPUT(vertices_all);
    CHECK_INPUT(textures);
    CHECK_INPUT(image);
    CHECK_FLOATING(vertices_all);
    CHECK_FLOATING(textures);
    CHECK_FLOATING(image);
    CHECK_SAME_DEVICE(vertices_all, textures);
    CHECK_SAME_DEVICE(vertices_all, image);
    CHECK_SAME_DTYPE(vertices_all, textures);
    CHECK_SAME_DTYPE(vertices_all, image);

    const c10::cuda::CUDAGuard device_guard(vertices_all.device());
    return create_texture_image_cuda(vertices_all, textures, image, eps);
}

PYBIND11_MODULE(create_texture_image, m) {
    m.def("create_texture_image", &create_texture_image, "CREATE_TEXTURE_IMAGE (CUDA)");
}

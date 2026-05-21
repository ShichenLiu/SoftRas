#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

// CUDA forward declarations

at::Tensor load_textures_cuda(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor is_update);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOATING(x) TORCH_CHECK(at::isFloatingType((x).scalar_type()), #x " must be a floating point tensor")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be an int32 tensor")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_SAME_DEVICE(x, y) TORCH_CHECK((x).device() == (y).device(), #x " and " #y " must be on the same device")
#define CHECK_SAME_DTYPE(x, y) TORCH_CHECK((x).scalar_type() == (y).scalar_type(), #x " and " #y " must have the same dtype")

at::Tensor load_textures(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor is_update) {

    CHECK_INPUT(image);
    CHECK_INPUT(faces);
    CHECK_INPUT(is_update);
    CHECK_INPUT(textures);
    CHECK_FLOATING(image);
    CHECK_FLOATING(faces);
    CHECK_FLOATING(textures);
    CHECK_INT32(is_update);
    CHECK_SAME_DEVICE(image, faces);
    CHECK_SAME_DEVICE(image, textures);
    CHECK_SAME_DEVICE(image, is_update);
    CHECK_SAME_DTYPE(image, faces);
    CHECK_SAME_DTYPE(image, textures);

    const c10::cuda::CUDAGuard device_guard(image.device());
    return load_textures_cuda(image, faces, textures, is_update);
}

PYBIND11_MODULE(load_textures, m) {
    m.def("load_textures", &load_textures, "LOAD_TEXTURES (CUDA)");
}

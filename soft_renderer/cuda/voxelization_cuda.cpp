#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> voxelize_sub1_cuda(
        at::Tensor faces,
        at::Tensor voxels);

std::vector<at::Tensor> voxelize_sub2_cuda(
        at::Tensor faces,
        at::Tensor voxels);

std::vector<at::Tensor> voxelize_sub3_cuda(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible);

std::vector<at::Tensor> voxelize_sub4_cuda(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOATING(x) TORCH_CHECK(at::isFloatingType((x).scalar_type()), #x " must be a floating point tensor")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be an int32 tensor")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_SAME_DEVICE(x, y) TORCH_CHECK((x).device() == (y).device(), #x " and " #y " must be on the same device")

std::vector<at::Tensor> voxelize_sub1(
        at::Tensor faces,
        at::Tensor voxels) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);
    CHECK_FLOATING(faces);
    CHECK_INT32(voxels);
    CHECK_SAME_DEVICE(faces, voxels);

    const c10::cuda::CUDAGuard device_guard(faces.device());
    return voxelize_sub1_cuda(faces, voxels);
}

std::vector<at::Tensor> voxelize_sub2(
        at::Tensor faces,
        at::Tensor voxels) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);
    CHECK_FLOATING(faces);
    CHECK_INT32(voxels);
    CHECK_SAME_DEVICE(faces, voxels);

    const c10::cuda::CUDAGuard device_guard(faces.device());
    return voxelize_sub2_cuda(faces, voxels);
}

std::vector<at::Tensor> voxelize_sub3(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);
    CHECK_INPUT(visible);
    CHECK_FLOATING(faces);
    CHECK_INT32(voxels);
    CHECK_INT32(visible);
    CHECK_SAME_DEVICE(faces, voxels);
    CHECK_SAME_DEVICE(faces, visible);

    const c10::cuda::CUDAGuard device_guard(faces.device());
    return voxelize_sub3_cuda(faces, voxels, visible);
}

std::vector<at::Tensor> voxelize_sub4(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);
    CHECK_INPUT(visible);
    CHECK_FLOATING(faces);
    CHECK_INT32(voxels);
    CHECK_INT32(visible);
    CHECK_SAME_DEVICE(faces, voxels);
    CHECK_SAME_DEVICE(faces, visible);

    const c10::cuda::CUDAGuard device_guard(faces.device());
    return voxelize_sub4_cuda(faces, voxels, visible);
}

PYBIND11_MODULE(voxelization, m) {
    m.def("voxelize_sub1", &voxelize_sub1, "VOXELIZE_SUB1 (CUDA)");
    m.def("voxelize_sub2", &voxelize_sub2, "VOXELIZE_SUB2 (CUDA)");
    m.def("voxelize_sub3", &voxelize_sub3, "VOXELIZE_SUB3 (CUDA)");
    m.def("voxelize_sub4", &voxelize_sub4, "VOXELIZE_SUB4 (CUDA)");
}

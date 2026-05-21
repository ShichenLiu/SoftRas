#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> forward_soft_rasterize_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor soft_colors,
        int image_size,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side);

std::vector<at::Tensor> backward_soft_rasterize_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor soft_colors,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor grad_faces,
        at::Tensor grad_textures,
        at::Tensor grad_soft_colors,
        int image_size,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOATING(x) TORCH_CHECK(at::isFloatingType((x).scalar_type()), #x " must be a floating point tensor")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_SAME_DEVICE(x, y) TORCH_CHECK((x).device() == (y).device(), #x " and " #y " must be on the same device")
#define CHECK_SAME_DTYPE(x, y) TORCH_CHECK((x).scalar_type() == (y).scalar_type(), #x " and " #y " must have the same dtype")

std::vector<at::Tensor> forward_soft_rasterize(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor soft_colors,
        int image_size,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(faces_info);
    CHECK_INPUT(aggrs_info);
    CHECK_INPUT(soft_colors);
    CHECK_FLOATING(faces);
    CHECK_FLOATING(textures);
    CHECK_FLOATING(faces_info);
    CHECK_FLOATING(aggrs_info);
    CHECK_FLOATING(soft_colors);
    CHECK_SAME_DEVICE(faces, textures);
    CHECK_SAME_DEVICE(faces, faces_info);
    CHECK_SAME_DEVICE(faces, aggrs_info);
    CHECK_SAME_DEVICE(faces, soft_colors);
    CHECK_SAME_DTYPE(faces, textures);
    CHECK_SAME_DTYPE(faces, faces_info);
    CHECK_SAME_DTYPE(faces, aggrs_info);
    CHECK_SAME_DTYPE(faces, soft_colors);

    const c10::cuda::CUDAGuard device_guard(faces.device());
    return forward_soft_rasterize_cuda(
        faces,
        textures,
        faces_info,
        aggrs_info,
        soft_colors,
        image_size,
        near,
        far,
        eps,
        sigma_val,
        func_id_dist,
        dist_eps,
        gamma_val,
        func_id_rgb,
        func_id_alpha,
        texture_sample_type,
        double_side);
}

std::vector<at::Tensor> backward_soft_rasterize(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor soft_colors,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor grad_faces,
        at::Tensor grad_textures,
        at::Tensor grad_soft_colors,
        int image_size,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(soft_colors);
    CHECK_INPUT(faces_info);
    CHECK_INPUT(aggrs_info);
    CHECK_INPUT(grad_faces);
    CHECK_INPUT(grad_textures);
    CHECK_INPUT(grad_soft_colors);
    CHECK_FLOATING(faces);
    CHECK_FLOATING(textures);
    CHECK_FLOATING(soft_colors);
    CHECK_FLOATING(faces_info);
    CHECK_FLOATING(aggrs_info);
    CHECK_FLOATING(grad_faces);
    CHECK_FLOATING(grad_textures);
    CHECK_FLOATING(grad_soft_colors);
    CHECK_SAME_DEVICE(faces, textures);
    CHECK_SAME_DEVICE(faces, soft_colors);
    CHECK_SAME_DEVICE(faces, faces_info);
    CHECK_SAME_DEVICE(faces, aggrs_info);
    CHECK_SAME_DEVICE(faces, grad_faces);
    CHECK_SAME_DEVICE(faces, grad_textures);
    CHECK_SAME_DEVICE(faces, grad_soft_colors);
    CHECK_SAME_DTYPE(faces, textures);
    CHECK_SAME_DTYPE(faces, soft_colors);
    CHECK_SAME_DTYPE(faces, faces_info);
    CHECK_SAME_DTYPE(faces, aggrs_info);
    CHECK_SAME_DTYPE(faces, grad_faces);
    CHECK_SAME_DTYPE(faces, grad_textures);
    CHECK_SAME_DTYPE(faces, grad_soft_colors);

    const c10::cuda::CUDAGuard device_guard(faces.device());
    return backward_soft_rasterize_cuda(
        faces,
        textures,
        soft_colors,
        faces_info,
        aggrs_info,
        grad_faces,
        grad_textures,
        grad_soft_colors,
        image_size,
        near,
        far,
        eps,
        sigma_val,
        func_id_dist,
        dist_eps,
        gamma_val,
        func_id_rgb,
        func_id_alpha,
        texture_sample_type,
        double_side);
}

PYBIND11_MODULE(soft_rasterize, m) {
    m.def("forward_soft_rasterize", &forward_soft_rasterize, "FORWARD_SOFT_RASTERIZE (CUDA)");
    m.def("backward_soft_rasterize", &backward_soft_rasterize, "BACKWARD_SOFT_RASTERIZE (CUDA)");
}

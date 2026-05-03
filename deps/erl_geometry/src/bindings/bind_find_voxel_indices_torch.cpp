#include "erl_common/pybind11.hpp"
#include "erl_geometry/find_voxel_indices_torch.hpp"

#ifdef ERL_USE_LIBTORCH
    #include <torch/extension.h>
#endif

void
BindFindVoxelIndicesTorch(py::module &m) {
#ifdef ERL_USE_LIBTORCH
    m.def(
        "find_voxel_indices",
        [](const torch::Tensor &codes,
           const int dims,
           const int n_levels,
           const torch::Tensor &children,
           const bool parallel) {
            torch::Tensor voxel_indices;
            erl::geometry::FindVoxelIndicesTorch(
                codes,
                dims,
                n_levels,
                children,
                parallel,
                voxel_indices);
            return voxel_indices;
        },
        py::arg("codes").noconvert(),
        py::arg("dims"),
        py::arg("n_levels"),
        py::arg("children").noconvert(),
        py::arg("parallel") = true,
        py::call_guard<py::gil_scoped_release>(),
        R"pbdoc(
Find voxel indices from morton codes and tree structure.

Args:
    codes (torch.Tensor): Tensor of morton code with dtype torch.uint32 or torch.uint64.
    dims (int): Space dimension, 2 or 3.
    n_levels (int): The number of levels to search from the root.
    children (torch.Tensor): Tensor of shape (M, 2^dims) that stores the tree structure.
    parallel (bool, optional): If true, use parallel for to speed up the search when using CPU. Default is True.

Returns:
    torch.Tensor: Tensor to store the found voxel indices, has the same shape as codes and the same dtype as children.)pbdoc");
#else
    (void) m;
#endif
}

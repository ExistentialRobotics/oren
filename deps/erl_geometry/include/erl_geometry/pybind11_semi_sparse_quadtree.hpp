#pragma once

#include "pybind11_quadtree_impl.hpp"
#include "semi_sparse_nd_tree_setting.hpp"
#include "semi_sparse_quadtree_base.hpp"

#ifdef ERL_USE_OPENCV
    #include "pybind11_semi_sparse_quadtree_drawer.hpp"
#endif

#ifdef ERL_USE_LIBTORCH
    #include <torch/extension.h>
#endif

template<class Quadtree, class Node>
auto
BindSemiSparseQuadtree(
    const py::module &m,
    const char *tree_name,
    std::function<void(py::class_<
                       Quadtree,
                       erl::geometry::AbstractQuadtree<typename Quadtree::DataType>,
                       std::shared_ptr<Quadtree>> &)> additional_bindings = nullptr) {

    using namespace erl::common;
    using namespace erl::geometry;
    using Dtype = typename Quadtree::DataType;
    using Matrix2X = Eigen::Matrix2X<Dtype>;

    py::class_<Quadtree, AbstractQuadtree<Dtype>, std::shared_ptr<Quadtree>> tree(m, tree_name);

    if (additional_bindings) { additional_bindings(tree); }
    if (std::is_same_v<typename Quadtree::Setting, SemiSparseNdTreeSetting> &&
        !py::hasattr(tree, "Setting")) {
        ERL_DEBUG("Bind default Setting type to {}", tree_name);
        tree.def("Setting", []() { return std::make_shared<typename Quadtree::Setting>(); });
    }

    // AbstractQuadtree methods are defined in bind_abstract_quadtree.cpp

    // SemiSparseQuadtree
    tree.def(py::init<>())
        .def(
            py::init<>([](const std::shared_ptr<typename Quadtree::Setting> &setting) {
                return std::make_shared<Quadtree>(setting);
            }),
            py::arg("setting"))
        .def(
            py::init<>(
                [](const std::string &filename) { return std::make_shared<Quadtree>(filename); }),
            py::arg("filename"))
        .def_property_readonly(
            "setting",
            &Quadtree::template GetSetting<typename Quadtree::Setting>)
        .def_property_readonly("parents", &Quadtree::GetParents)
        .def_property_readonly("children", &Quadtree::GetChildren)
        .def_property_readonly("voxels", &Quadtree::GetVoxels)
        .def_property_readonly("voxel_centers", &Quadtree::GetVoxelCenters)
        .def_property_readonly("vertices", &Quadtree::GetVertices)
        .def_property_readonly("num_vertices", &Quadtree::GetVertexCount)
        .def_property_readonly(
            "num_independent_leaf_vertices",
            &Quadtree::GetIndependentLeafVertexCount)
        .def_property_readonly("vertex_keys", &Quadtree::GetVertexKeys)
        .def(
            "insert_points",
            py::overload_cast<const Matrix2X &>(&Quadtree::InsertPoints),
            py::arg("points"))
        .def(
            "insert_keys",
            [](Quadtree &self, const Eigen::Matrix2X<QuadtreeKey::KeyType> &keys) {
                auto keys_ptr = reinterpret_cast<const QuadtreeKey *>(keys.data());
                self.InsertKeys(keys_ptr, keys.cols());
            },
            py::arg("keys"))
        .def("insert_key", &Quadtree::InsertKey, py::arg("key"), py::arg("max_depth"))
        .def(
            "find_voxel_indices",
            py::overload_cast<const Matrix2X &, bool>(&Quadtree::FindVoxelIndices, py::const_),
            py::arg("points"),
            py::arg("parallel"),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "find_voxel_indices",
            [](const Quadtree &self,
               const Eigen::Matrix2X<QuadtreeKey::KeyType> &keys,
               bool parallel) {
                auto keys_ptr = reinterpret_cast<const QuadtreeKey *>(keys.data());
                return self.FindVoxelIndices(keys_ptr, keys.cols(), parallel);
            },
            py::arg("keys"),
            py::arg("parallel"))
        .def("find_voxel_index", &Quadtree::FindVoxelIndex, py::arg("key"));

#ifdef ERL_USE_LIBTORCH
    // extra bindings to support libtorch tensors
    tree.def_property_readonly(
            "parents_tensor",
            [](const Quadtree &self) {
                const auto &parents = self.GetParents();
                torch::Tensor parents_tensor =
                    torch::from_blob(
                        const_cast<typename Quadtree::NodeIndex *>(parents.data()),
                        {static_cast<long>(parents.size())},
                        c10::CppTypeToScalarType<typename Quadtree::NodeIndex>::value)
                        .clone();
                return parents_tensor;
            })
        .def_property_readonly(
            "children_tensor",
            [](const Quadtree &self) {
                const auto &children = self.GetChildren();
                torch::Tensor children_tensor =
                    torch::from_blob(
                        const_cast<typename Quadtree::NodeIndex *>(children.data()),
                        {static_cast<long>(children.cols()), 4},
                        c10::CppTypeToScalarType<typename Quadtree::NodeIndex>::value)
                        .clone();
                return children_tensor;
            })
        .def_property_readonly(
            "voxels_tensor",
            [](const Quadtree &self) {
                const auto &voxels = self.GetVoxels();
                torch::Tensor voxels_tensor =
                    torch::from_blob(
                        const_cast<QuadtreeKey::KeyType *>(voxels.data()),
                        {static_cast<long>(voxels.cols()), 3},  // (x, y, size)
                        c10::CppTypeToScalarType<QuadtreeKey::KeyType>::value)
                        .clone();
                return voxels_tensor;
            })
        .def_property_readonly(
            "voxel_centers_tensor",
            [](const Quadtree &self) {
                const auto &voxel_centers = self.GetVoxelCenters();
                torch::Tensor voxel_centers_tensor =
                    torch::from_blob(
                        const_cast<Dtype *>(voxel_centers.data()),
                        {static_cast<long>(voxel_centers.cols()), 2},  // (x, y)
                        c10::CppTypeToScalarType<Dtype>::value)
                        .clone();
                return voxel_centers_tensor;
            })
        .def_property_readonly(
            "vertices_tensor",
            [](const Quadtree &self) {
                const auto &vertices = self.GetVertices();
                torch::Tensor vertices_tensor =
                    torch::from_blob(
                        const_cast<typename Quadtree::NodeIndex *>(vertices.data()),
                        {static_cast<long>(vertices.cols()), 4},
                        c10::CppTypeToScalarType<typename Quadtree::NodeIndex>::value)
                        .clone();
                return vertices_tensor;
            })
        .def(
            "insert_points",
            [](Quadtree &self, const torch::Tensor &points) {
                ERL_ASSERTM(points.is_floating_point(), "points must be floating point tensor.");
                ERL_ASSERTM(points.ndimension() >= 2, "points must be (..., 2) tensor.");
                ERL_ASSERTM(
                    points.size(points.ndimension() - 1) == 2,
                    "points must be (..., 2) tensor.");

                const auto setting = self.template GetSetting<typename Quadtree::Setting>();
                const uint32_t key_offset = 1 << (setting->tree_depth - 1);
                torch::Tensor keys = (points / setting->resolution + key_offset).cpu();
                keys = keys.to(c10::CppTypeToScalarType<QuadtreeKey::KeyType>::value).contiguous();

                const auto key_val_ptr = keys.const_data_ptr<QuadtreeKey::KeyType>();
                const auto key_ptr = reinterpret_cast<const QuadtreeKey *>(key_val_ptr);
                const long num_keys = keys.numel() / 2;

                auto indices = self.InsertKeys(key_ptr, num_keys);
                torch::Tensor indices_tensor =
                    torch::from_blob(
                        indices.data(),
                        keys.sizes().slice(0, keys.ndimension() - 1),
                        c10::CppTypeToScalarType<typename Quadtree::NodeIndex>::value)
                        .clone()
                        .to(keys.device());
                return indices_tensor;
            },
            py::arg("points"),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "insert_keys",
            [](Quadtree &self, const torch::Tensor &keys) {
                ERL_ASSERTM(!keys.is_floating_point(), "keys must be integer tensor.");
                ERL_ASSERTM(keys.ndimension() >= 2, "keys must be (..., 2) tensor.");
                ERL_ASSERTM(keys.size(keys.ndimension() - 1) == 2, "keys must be (..., 2) tensor.");

                const torch::Tensor keys_ =
                    keys.cpu()
                        .to(c10::CppTypeToScalarType<QuadtreeKey::KeyType>::value)
                        .contiguous();

                const auto key_val_ptr = keys_.const_data_ptr<QuadtreeKey::KeyType>();
                const auto key_ptr = reinterpret_cast<const QuadtreeKey *>(key_val_ptr);
                const long num_keys = keys_.numel() / 2;

                auto indices = self.InsertKeys(key_ptr, num_keys);
                torch::Tensor indices_tensor =
                    torch::from_blob(
                        indices.data(),
                        keys.sizes().slice(0, keys.ndimension() - 1),
                        c10::CppTypeToScalarType<typename Quadtree::NodeIndex>::value)
                        .clone()
                        .to(keys.device());
                return indices_tensor;
            },
            py::arg("keys"),
            py::call_guard<py::gil_scoped_release>());
#endif

    BindQuadtreeImpl<decltype(tree), Dtype, Quadtree, Node>(tree);

#ifdef ERL_USE_OPENCV
    BindSemiSparseQuadtreeDrawer<Quadtree>(tree, "Drawer");

    tree.def(
        "visualize",
        [](std::shared_ptr<Quadtree> &self,
           const bool leaf_only,
           std::optional<Eigen::Vector2f> area_min,
           std::optional<Eigen::Vector2f> area_max,
           const Dtype resolution,
           const int padding,
           Eigen::Vector4i bg_color,
           Eigen::Vector4i fg_color,
           Eigen::Vector4i border_color,
           const int border_thickness) {
            using Drawer = SemiSparseQuadtreeDrawer<Quadtree>;
            auto drawer_setting = std::make_shared<typename Drawer::Setting>();
            if (area_min.has_value()) {
                drawer_setting->area_min = area_min.value();
            } else {
                Dtype min_x, min_y;
                self->GetMetricMin(min_x, min_y);
                drawer_setting->area_min[0] = min_x;
                drawer_setting->area_min[1] = min_y;
            }
            if (area_max.has_value()) {
                drawer_setting->area_max = area_max.value();
            } else {
                Dtype max_x, max_y;
                self->GetMetricMax(max_x, max_y);
                drawer_setting->area_max[0] = max_x;
                drawer_setting->area_max[1] = max_y;
            }
            drawer_setting->resolution = resolution;
            drawer_setting->padding = padding;
            for (int i = 0; i < 4; ++i) {
                drawer_setting->bg_color[i] = bg_color[i];
                drawer_setting->fg_color[i] = fg_color[i];
                drawer_setting->border_color[i] = border_color[i];
            }
            drawer_setting->border_thickness = border_thickness;

            auto drawer = std::make_shared<Drawer>(drawer_setting, self);
            cv::Mat mat;
            if (leaf_only) {
                drawer->DrawLeaves(mat);
            } else {
                drawer->DrawTree(mat);
            }
            cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGBA);
            Eigen::MatrixX8U image;
            cv::cv2eigen(mat, image);
            return image;
        },
        py::arg("leaf_only") = false,
        py::arg("area_min") = py::none(),
        py::arg("area_max") = py::none(),
        py::arg("resolution") = 0.1,
        py::arg("padding") = 1,
        py::arg("bg_color") = Eigen::Vector4i(128, 128, 128, 255),
        py::arg("fg_color") = Eigen::Vector4i(255, 255, 255, 255),
        py::arg("border_color") = Eigen::Vector4i(0, 0, 0, 255),
        py::arg("border_thickness") = 1);
#endif

    return tree;
}

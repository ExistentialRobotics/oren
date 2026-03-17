#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstring>

#include <torch/torch.h>

#include "grad_sdf_cpp/sdf_network_module.hpp"

// Simple NPY file format writer for 3D float32 grids
void SaveNumpyFloat32Grid(const std::string &path, const torch::Tensor &tensor)
{
    if (tensor.dtype() != torch::kFloat32)
    {
        throw std::runtime_error("Only float32 tensors are supported");
    }
    if (tensor.dim() != 3)
    {
        throw std::runtime_error("Only 3D tensors are supported");
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    // Write magic number
    file.write("\x93NUMPY", 6);

    // Write version 1.0
    uint8_t major = 1, minor = 0;
    file.write(reinterpret_cast<const char *>(&major), 1);
    file.write(reinterpret_cast<const char *>(&minor), 1);

    // Build header for 3D array
    auto sizes = tensor.sizes();
    std::string shape_str = "(";
    for (int i = 0; i < 3; ++i)
    {
        if (i > 0)
            shape_str += ", ";
        shape_str += std::to_string(sizes[i]);
    }
    shape_str += ")";

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
    // Pad to multiple of 16
    while ((header.length() + 1) % 16 != 0)
    {
        header += " ";
    }
    header += "\n";

    uint16_t header_len = static_cast<uint16_t>(header.length());
    file.write(reinterpret_cast<const char *>(&header_len), 2);
    file.write(header.c_str(), header.length());

    // Write data
    auto cpu_tensor = tensor.to(torch::kCPU).contiguous();
    auto *data_ptr = cpu_tensor.data_ptr<float>();
    auto numel = cpu_tensor.numel();
    file.write(reinterpret_cast<const char *>(data_ptr), numel * sizeof(float));
}

int main(int argc, char *argv[])
{
    try
    {
        using namespace grad_sdf_cpp;

        // Parse command line arguments
        std::string config_path, bundle_path, device_str = "cpu", output_dir = ".";
        std::vector<float> bound_min = {-2.0f, -2.0f, 0.0f};
        std::vector<float> bound_max = {2.0f, 2.0f, 2.0f};
        bool has_bound_min_override = false;
        bool has_bound_max_override = false;
        float grid_resolution = 0.05f;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--config" && i + 1 < argc)
            {
                config_path = argv[++i];
            }
            else if (arg == "--bundle" && i + 1 < argc)
            {
                bundle_path = argv[++i];
            }
            else if (arg == "--device" && i + 1 < argc)
            {
                device_str = argv[++i];
            }
            else if (arg == "--bound-min" && i + 3 < argc)
            {
                bound_min[0] = std::stof(argv[++i]);
                bound_min[1] = std::stof(argv[++i]);
                bound_min[2] = std::stof(argv[++i]);
                has_bound_min_override = true;
            }
            else if (arg == "--bound-max" && i + 3 < argc)
            {
                bound_max[0] = std::stof(argv[++i]);
                bound_max[1] = std::stof(argv[++i]);
                bound_max[2] = std::stof(argv[++i]);
                has_bound_max_override = true;
            }
            else if (arg == "--grid-resolution" && i + 1 < argc)
            {
                grid_resolution = std::stof(argv[++i]);
            }
            else if (arg == "--output" && i + 1 < argc)
            {
                output_dir = argv[++i];
            }
        }

        if (config_path.empty())
        {
            throw std::runtime_error("--config is required");
        }
        if (bundle_path.empty())
        {
            throw std::runtime_error("--bundle is required");
        }

        torch::Device device(device_str);

        // Load config
        auto cfg = grad_sdf_cpp::LoadSdfNetworkConfigFromTrainerYaml(config_path);
        const auto &cfg_bound_min = cfg.residual_net_cfg.bound_min;
        const auto &cfg_bound_max = cfg.residual_net_cfg.bound_max;

        // Default to config bounds (dataset_args overrides are already applied in config loader)
        if (!has_bound_min_override && cfg_bound_min.size() == 3)
        {
            bound_min = {
                static_cast<float>(cfg_bound_min[0]),
                static_cast<float>(cfg_bound_min[1]),
                static_cast<float>(cfg_bound_min[2]),
            };
        }
        if (!has_bound_max_override && cfg_bound_max.size() == 3)
        {
            bound_max = {
                static_cast<float>(cfg_bound_max[0]),
                static_cast<float>(cfg_bound_max[1]),
                static_cast<float>(cfg_bound_max[2]),
            };
        }

        std::cout << "Extracting SDF grid\n";
        std::cout << "  Config: " << config_path << "\n";
        std::cout << "  Bundle: " << bundle_path << "\n";
        std::cout << "  Bounds: [" << bound_min[0] << ", " << bound_min[1] << ", " << bound_min[2] << "] to ["
                  << bound_max[0] << ", " << bound_max[1] << ", " << bound_max[2] << "]\n";
        std::cout << "  Resolution: " << grid_resolution << "\n";
        std::cout << "  Device: " << device_str << "\n";
        std::cout << "  Output: " << output_dir << "\n";

        // Load model
        std::cout << "\nLoading model from bundle " << bundle_path << "\n";
        auto model = grad_sdf_cpp::SdfNetworkModule(cfg);
        model->to(device);
        model->eval();
        model->LoadFromBundle(bundle_path);

        // Compute grid dimensions
        int nx = static_cast<int>(std::ceil((bound_max[0] - bound_min[0]) / grid_resolution)) + 1;
        int ny = static_cast<int>(std::ceil((bound_max[1] - bound_min[1]) / grid_resolution)) + 1;
        int nz = static_cast<int>(std::ceil((bound_max[2] - bound_min[2]) / grid_resolution)) + 1;

        std::cout << "Grid dimensions: " << nx << " x " << ny << " x " << nz << " = " << (nx * ny * nz)
                  << " points\n";

        // Create grid points
        std::cout << "\nGenerating " << (nx * ny * nz) << " grid points...\n";
        std::vector<torch::Tensor> grid_points_list;
        std::vector<int64_t> point_dims = {nx, ny, nz, 3};

        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int k = 0; k < nz; ++k)
                {
                    std::vector<float> pt = {
                        bound_min[0] + i * grid_resolution,
                        bound_min[1] + j * grid_resolution,
                        bound_min[2] + k * grid_resolution,
                    };
                    grid_points_list.push_back(torch::tensor(pt, torch::kFloat32));
                }
            }
        }

        auto grid_points = torch::stack(grid_points_list).to(device);
        grid_points = grid_points.reshape({nx, ny, nz, 3});

        // Evaluate model on grid
        std::cout << "Evaluating model on grid...\n";
        torch::NoGradGuard no_grad;
        auto sdf_values = model->forward(grid_points.reshape({-1, 3})).sdf;
        sdf_values = sdf_values.reshape({nx, ny, nz});

        std::cout << "  SDF shape: " << sdf_values.sizes() << "\n";
        std::cout << "  SDF range: [" << sdf_values.min().item<float>() << ", " << sdf_values.max().item<float>()
                  << "]\n";

        // Save grid
        std::string output_file = output_dir + "/sdf_grid.npy";
        std::cout << "\nSaving SDF grid to " << output_file << "\n";
        SaveNumpyFloat32Grid(output_file, sdf_values);

        // Also save grid metadata
        std::string metadata_file = output_dir + "/grid_metadata.txt";
        std::ofstream metadata(metadata_file);
        metadata << "nx=" << nx << "\n";
        metadata << "ny=" << ny << "\n";
        metadata << "nz=" << nz << "\n";
        metadata << "grid_resolution=" << grid_resolution << "\n";
        metadata << "bound_min=" << bound_min[0] << "," << bound_min[1] << "," << bound_min[2] << "\n";
        metadata << "bound_max=" << bound_max[0] << "," << bound_max[1] << "," << bound_max[2] << "\n";
        metadata.close();
        std::cout << "Saved metadata to " << metadata_file << "\n";

        std::cout << "\n✓ Grid extraction complete\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

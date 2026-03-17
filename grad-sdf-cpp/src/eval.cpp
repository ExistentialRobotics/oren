#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "grad_sdf_cpp/config.hpp"
#include "grad_sdf_cpp/sdf_network_module.hpp"

namespace
{

    struct Args
    {
        std::string config_path;
        std::string bundle_dir;
        std::string device = "cpu";
        std::string points_file;
        std::string output_file;
    };

    void PrintUsage()
    {
        std::cout
            << "Usage:\n"
            << "  grad_sdf_eval --config <trainer.yaml> --bundle <cpp_bundle_dir>\n"
            << "                --points <points.npy> --output <output.npy>\n"
            << "                [--device cpu|cuda]\n";
    }

    Args ParseArgs(int argc, char **argv)
    {
        Args args;

        for (int i = 1; i < argc; ++i)
        {
            const std::string key = argv[i];
            if (key == "--config" && i + 1 < argc)
            {
                args.config_path = argv[++i];
            }
            else if (key == "--bundle" && i + 1 < argc)
            {
                args.bundle_dir = argv[++i];
            }
            else if (key == "--device" && i + 1 < argc)
            {
                args.device = argv[++i];
            }
            else if (key == "--points" && i + 1 < argc)
            {
                args.points_file = argv[++i];
            }
            else if (key == "--output" && i + 1 < argc)
            {
                args.output_file = argv[++i];
            }
            else if (key == "--help" || key == "-h")
            {
                PrintUsage();
                std::exit(0);
            }
            else
            {
                throw std::runtime_error("Unknown or incomplete argument: " + key);
            }
        }

        if (args.config_path.empty())
        {
            throw std::runtime_error("--config is required.");
        }
        if (args.bundle_dir.empty())
        {
            throw std::runtime_error("--bundle is required.");
        }

        return args;
    }

    // Simple NPY file format reader for float32
    torch::Tensor LoadNumpyFloat32(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + path);
        }

        // Read magic number
        char magic[6];
        file.read(magic, 6);
        if (std::string(magic, 6) != "\x93NUMPY")
        {
            throw std::runtime_error("Invalid NPY file magic number");
        }

        // Read version
        uint8_t major, minor;
        file.read(reinterpret_cast<char *>(&major), 1);
        file.read(reinterpret_cast<char *>(&minor), 1);

        // Read header length
        uint16_t header_len;
        if (major == 1)
        {
            file.read(reinterpret_cast<char *>(&header_len), 2);
        }
        else if (major == 3)
        {
            uint32_t header_len_32;
            file.read(reinterpret_cast<char *>(&header_len_32), 4);
            header_len = header_len_32;
        }
        else
        {
            throw std::runtime_error("Unsupported NPY version");
        }

        // Read and parse header (simplified - assumes dtype is float32)
        std::string header(header_len, ' ');
        file.read(&header[0], header_len);

        // Extract shape from header (very simplified parser)
        // For now, just read remaining data as float32
        std::vector<float> data;
        float val;
        while (file.read(reinterpret_cast<char *>(&val), sizeof(float)))
        {
            data.push_back(val);
        }

        if (data.empty())
        {
            throw std::runtime_error("No data in NPY file");
        }

        // Assume shape is (N, 3) for points
        int64_t num_points = data.size() / 3;
        if (data.size() % 3 != 0)
        {
            throw std::runtime_error("Data size is not divisible by 3");
        }

        auto tensor = torch::from_blob(
            data.data(),
            {num_points, 3},
            torch::TensorOptions().dtype(torch::kFloat32));
        return tensor.clone();
    }

    // Simple NPY file format writer for float32
    void SaveNumpyFloat32(const std::string &path, const torch::Tensor &tensor)
    {
        if (tensor.dtype() != torch::kFloat32)
        {
            throw std::runtime_error("Only float32 tensors are supported");
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

        // Build header
        std::string shape_str = "(";
        auto sizes = tensor.sizes();
        for (size_t i = 0; i < sizes.size(); ++i)
        {
            if (i > 0)
                shape_str += ", ";
            shape_str += std::to_string(sizes[i]);
        }
        if (sizes.size() == 1)
            shape_str += ","; // Trailing comma for 1D numpy arrays
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

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const auto args = ParseArgs(argc, argv);

        std::cout << "Loading configuration from " << args.config_path << "\n";
        auto cfg = grad_sdf_cpp::LoadSdfNetworkConfigFromTrainerYaml(args.config_path);

        std::cout << "Loading points from " << args.points_file << "\n";
        auto points = LoadNumpyFloat32(args.points_file);
        std::cout << "  Points shape: " << points.sizes() << "\n";

        const torch::Device device(args.device);
        points = points.to(device);

        std::cout << "Loading model from bundle " << args.bundle_dir << "\n";
        auto model = grad_sdf_cpp::SdfNetworkModule(cfg);
        model->to(device);
        model->eval();
        model->LoadFromBundle(args.bundle_dir);

        std::cout << "Evaluating " << points.size(0) << " points...\n";
        torch::NoGradGuard no_grad;
        auto out = model->forward(points);

        std::cout << "Saving output to " << args.output_file << "\n";
        SaveNumpyFloat32(args.output_file, out.sdf.to(torch::kCPU));

        std::cout << "\n✓ Evaluation complete\n";
        std::cout << "  Output shape: " << out.sdf.sizes() << "\n";
        std::cout << "  Output dtype: " << out.sdf.dtype() << "\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        PrintUsage();
        return 1;
    }
}

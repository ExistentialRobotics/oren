#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include <torch/torch.h>

#include "grad_sdf_cpp/config.hpp"
#include "grad_sdf_cpp/sdf_network_module.hpp"

namespace
{

#ifdef GRAD_SDF_CPP_SOURCE_DIR
    const std::string kDefaultConfigPath = std::string(GRAD_SDF_CPP_SOURCE_DIR) + "/models/trainer-ros.yaml";
    const std::string kDefaultBundleDir = std::string(GRAD_SDF_CPP_SOURCE_DIR) + "/models/bundle";
#else
    const std::string kDefaultConfigPath = "models/trainer-ros.yaml";
    const std::string kDefaultBundleDir = "models/bundle";
#endif

    struct Args
    {
        std::string config_path = kDefaultConfigPath;
        std::string bundle_dir = kDefaultBundleDir;
        std::string device = "cuda";
        int64_t num_points = 4096;
        int64_t seed = 0;
        bool sample_only = false;
    };

    void PrintUsage()
    {
        std::cout
            << "Usage:\n"
            << "  grad_sdf_infer [--config <trainer.yaml>] [--bundle <cpp_bundle_dir>] [--device cpu|cuda]"
            << " [--num-points N] [--seed N] [--sample-only]\n"
            << "Defaults:\n"
            << "  --config " << kDefaultConfigPath << "\n"
            << "  --bundle " << kDefaultBundleDir << "\n"
            << "  --device cuda\n";
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
            else if (key == "--num-points" && i + 1 < argc)
            {
                args.num_points = std::stoll(argv[++i]);
            }
            else if (key == "--seed" && i + 1 < argc)
            {
                args.seed = std::stoll(argv[++i]);
            }
            else if (key == "--sample-only")
            {
                args.sample_only = true;
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

        if (!args.sample_only && args.bundle_dir.empty())
        {
            throw std::runtime_error("--bundle is required unless --sample-only is used.");
        }

        return args;
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const auto args = ParseArgs(argc, argv);

        auto cfg = grad_sdf_cpp::LoadSdfNetworkConfigFromTrainerYaml(args.config_path);
        torch::manual_seed(args.seed);

        const torch::Device device(args.device);

        std::vector<double> bound_min = cfg.residual_net_cfg.bound_min;
        std::vector<double> bound_max = cfg.residual_net_cfg.bound_max;
        if (bound_min.size() != 3 || bound_max.size() != 3)
        {
            bound_min = {-1.0, -1.0, -1.0};
            bound_max = {1.0, 1.0, 1.0};
        }

        auto points = torch::rand({args.num_points, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        auto bound_min_t = torch::tensor(bound_min, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        auto bound_max_t = torch::tensor(bound_max, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        points = points * (bound_max_t - bound_min_t) + bound_min_t;

        auto mins = points.amin(0).to(torch::kCPU);
        auto maxs = points.amax(0).to(torch::kCPU);
        const int64_t point_preview = std::min<int64_t>(5, points.size(0));
        auto points_preview = points.slice(0, 0, point_preview).to(torch::kCPU);

        std::cout << "Sampling complete\n";
        std::cout << "  points shape: " << points.sizes() << "\n";
        std::cout << "  sampled min xyz: " << mins << "\n";
        std::cout << "  sampled max xyz: " << maxs << "\n";
        std::cout << "  first " << point_preview << " points:\n"
                  << points_preview << "\n";

        if (args.sample_only)
        {
            std::cout << "Sample-only mode enabled; skipping model loading and forward pass.\n";
            return 0;
        }

        auto model = grad_sdf_cpp::SdfNetworkModule(cfg);
        model->to(device);
        model->eval();
        model->LoadFromBundle(args.bundle_dir);

        torch::NoGradGuard no_grad;
        auto out = model->forward(points);

        std::cout << "Forward pass complete\n";
        std::cout << "  sdf shape: " << out.sdf.sizes() << "\n";

        const int64_t preview = std::min<int64_t>(10, out.sdf.numel());
        auto sdf_preview = out.sdf.reshape({-1}).slice(0, 0, preview).to(torch::kCPU);
        std::cout << "  first " << preview << " sdf values: " << sdf_preview << "\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        PrintUsage();
        return 1;
    }
}

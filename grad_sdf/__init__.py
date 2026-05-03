import os

use_erl_geometry = os.getenv("USE_ERL_GEOMETRY", "1")  # default to use erl_geometry if available
use_erl_geometry = use_erl_geometry.lower() in ("1", "true", "yes")

if use_erl_geometry:
    try:
        import numpy as np
        from erl_geometry import MarchingCubes, torch

        # import open3d after erl_geometry to avoid potential conflicts in C++ extensions
        import open3d as o3d

        erl_geometry_loaded = True
    except ImportError as e:
        print(f"Failed to import erl_geometry. Please ensure it is installed correctly: {e}")
        print("Will use fallback implementations.")
        print("However, some features may be missing or have reduced performance.")

        erl_geometry_loaded = False
        import numpy as np
        import open3d as o3d
        import torch

        from grad_sdf.utils.mcubes_wrapper import MarchingCubesWrapper as MarchingCubes
else:
    erl_geometry_loaded = False
    import numpy as np
    import open3d as o3d
    import torch

    from grad_sdf.utils.mcubes_wrapper import MarchingCubesWrapper as MarchingCubes

# GUI and rendering modules from Open3D
# noinspection PyPep8
from open3d.visualization import gui as o3d_gui
from open3d.visualization import rendering as o3d_rendering

__all__ = [
    "erl_geometry_loaded",
    "MarchingCubes",
    "torch",
    "o3d",
    "np",
    "o3d_gui",
    "o3d_rendering",
]

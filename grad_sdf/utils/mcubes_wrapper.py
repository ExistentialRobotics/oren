import numpy as np
import open3d as o3d
import skimage
from colorama import Fore, Style
from tqdm import tqdm


class MarchingCubesWrapper:
    """
    A wrapper for skimage.measure.marching_cubes to have the same interface as erl_geometry.MarchingCubes.
    """

    @staticmethod
    def run(coords_min, grid_res, grid_shape, grid_values, mask, iso_value, *args, **kwargs):
        if mask is not None:
            tqdm.write(
                Fore.RED + "Using fallback MarchingCubes implementation from skimage. The mesh quality is lower than"
                " erl_geometry.MarchingCubes. Please install erl_geometry for better performance and more accurate"
                " results." + Style.RESET_ALL
            )

        assert len(grid_res) == 3
        assert grid_values.size == grid_shape[0] * grid_shape[1] * grid_shape[2]
        if mask is not None:
            assert mask.size == grid_values.size
            mask = mask.reshape(grid_shape).astype(bool)

        grid_values = grid_values.reshape(grid_shape)

        vertices, triangles, vertex_normals, _ = skimage.measure.marching_cubes(
            volume=grid_values,
            level=iso_value,
            spacing=grid_res,
            allow_degenerate=False,
            mask=mask,
        )

        vertices = vertices.astype(np.float64) + np.array(coords_min, dtype=np.float64)
        triangles = triangles.astype(np.int32)
        vertex_normals = vertex_normals.astype(np.float64)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
        mesh.compute_triangle_normals()
        triangle_normals = np.asarray(mesh.triangle_normals)

        return vertices.T, triangles.T, triangle_normals.T

#!/usr/bin/env python3
"""
Multi-mesh visualization tool with synchronized camera views.
This script allows comparing multiple meshes side-by-side with synchronized camera controls.

Requirements:
    - open3d
    - numpy

Usage:
    python mesh_vis_comp_replica.py

Features:
    - Synchronized camera views across multiple windows
    - Load/save camera configurations from JSON files
    - Enhanced lighting with Phong shading for depth perception
    - Backface culling for better performance
    - Dark background for maximum contrast and shadow visibility

Camera Configuration:
    - Place "view_status_o3d.json" in the script directory to auto-load camera view
    - Press 'Ctrl+C' during visualization to save current camera view
    - Saved files: "view_status_o3d_saved_YYYYMMDD_HHMMSS.json"

Controls:
    Mouse:
        - Left drag: Rotate camera
        - Right drag: Pan camera
        - Scroll: Zoom in/out
    Keyboard:
        - C: Save current camera view
        - P: Save screenshots of all windows
        - R: Reset camera
        - S: Toggle synchronization
        - L: Toggle lighting
        - H: Help
        - Q/ESC: Quit

Note: If you encounter import errors with dash/plotly, you may need to install dash:
    pip install dash --user
    or use a virtual environment with the required dependencies.

Command-line Arguments:
    --camera-config PATH    Path to camera configuration JSON file
    --help                  Show help message and exit

Examples:
    # Use default meshes and camera config
    python mesh_vis_comp_replica.py

    # Use custom camera configuration
    python mesh_vis_comp_replica.py --camera-config my_camera.json
"""

import json
import os
import pathlib
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class ViewerConfig(ConfigABC):
    mesh_files: List[str] = field(default_factory=list)
    mesh_labels: List[str] = field(default_factory=list)
    window_name: str = "Mesh Comparison"
    enable_enhanced_lighting: bool = True
    backface_culling: bool = True
    camera_config_file: Optional[str] = None
    window_width: int = 800
    window_height: int = 600
    n_rows: int = 0  # 0 = auto-calculate
    n_cols: int = 0  # 0 = auto-calculate
    output_dir: str = "viewer_output"
    background_color: List[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])  # RGB values [0-1]
    ignore_mesh_color: bool = False  # If true, ignore mesh vertex colors and use default coloring
    crop_bounds_file: Optional[str] = None  # Path to yaml file with crop bounds

    # Error coloring configuration
    ground_truth_mesh_idx: int = -1  # Index of mesh to use as ground truth (-1 = none)
    ground_truth_file: Optional[str] = None  # Path to ground truth mesh/point cloud
    enable_error_coloring: bool = False  # Enable error-based coloring with jet colormap
    gt_sample_points: int = 100000  # Number of points to sample from ground truth
    error_min: float = 0.0  # Minimum error for color mapping (meters)
    error_max: float = 0.1  # Maximum error for color mapping (meters)
    show_error_colorbar: bool = True  # Print error statistics to console


class SynchronizedMeshViewer:
    """
    A class to visualize multiple meshes with synchronized camera views.
    All viewers share the same camera parameters (view matrix, field of view, etc.)
    """

    def __init__(self, config: ViewerConfig):
        """
        Initialize the synchronized mesh viewer.

        Args:
            config: ViewerConfig object containing all configuration parameters
        """
        self.config = config
        if self.config.enable_error_coloring:
            self.config.ignore_mesh_color = False

        # Build mesh_files list from config
        if config.mesh_files and config.mesh_labels:
            if len(config.mesh_files) != len(config.mesh_labels):
                raise ValueError("mesh_files and mesh_labels must have the same length")
            self.mesh_files = list(zip(config.mesh_labels, config.mesh_files))
        else:
            self.mesh_files = []

        self.window_name = config.window_name
        self.enable_enhanced_lighting = config.enable_enhanced_lighting
        self.camera_config_file = config.camera_config_file
        self.window_width = config.window_width
        self.window_height = config.window_height

        self.visualizers = []
        self.meshes = []
        self.view_controls = []
        self.is_running = True
        self.sync_lock = threading.Lock()
        self.reference_view_control = None
        self.camera_config = None

        # Camera synchronization parameters
        self.last_camera_params = None
        self.sync_enabled = True

    def parse_background_color(self, color_input) -> np.ndarray:
        """
        Parse background color from various input formats.

        Args:
            color_input: Can be:
                - List/tuple of 3 floats [0-1]: [R, G, B]
                - String name: "black", "white", "gray", "dark_gray"
                - None: Uses default [0.05, 0.05, 0.05]

        Returns:
            numpy array of [R, G, B] values in range [0-1]
        """
        # Predefined color names
        color_presets = {
            "black": [0.0, 0.0, 0.0],
            "dark_gray": [0.05, 0.05, 0.05],
            "gray": [0.5, 0.5, 0.5],
            "light_gray": [0.8, 0.8, 0.8],
            "white": [1.0, 1.0, 1.0],
            "dark_blue": [0.0, 0.0, 0.1],
            "dark_green": [0.0, 0.1, 0.0],
        }

        if color_input is None:
            return np.array([0.05, 0.05, 0.05])

        # Check if it's a string preset
        if isinstance(color_input, str):
            if color_input.lower() in color_presets:
                return np.array(color_presets[color_input.lower()])
            else:
                print(f"  Warning: Unknown color preset '{color_input}', using default dark_gray")
                return np.array([0.05, 0.05, 0.05])

        # Check if it's a list/tuple of RGB values
        if isinstance(color_input, (list, tuple)):
            if len(color_input) == 3:
                # Validate range [0-1]
                if all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in color_input):
                    return np.array(color_input)
                else:
                    print(f"  Warning: Background color values must be in range [0-1], got {color_input}")
                    return np.array([0.05, 0.05, 0.05])
            else:
                print(f"  Warning: Background color must have 3 values (RGB), got {len(color_input)}")
                return np.array([0.05, 0.05, 0.05])

        print(f"  Warning: Invalid background_color format {color_input}, using default")
        return np.array([0.05, 0.05, 0.05])

    def load_crop_bounds(self) -> Optional[o3d.geometry.OrientedBoundingBox]:
        if self.config.crop_bounds_file is None:
            return None

        try:
            with open(self.config.crop_bounds_file, "r") as f:
                bounds = yaml.safe_load(f)
            size = np.array(bounds["size"])
            rotation = np.array(bounds["rotation"])  # w, x, y, z
            center = np.array(bounds["center"])
            print(f"Loaded crop bounds from: {self.config.crop_bounds_file}")

            obb = o3d.geometry.OrientedBoundingBox(
                center=center,
                R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_quaternion(rotation),
                extent=size,
            )
            return obb
        except Exception as e:
            print(f"Error loading crop bounds: {e}")
            return None

    def load_meshes(self) -> bool:
        """
        Load all mesh files.

        Returns:
            True if all meshes loaded successfully, False otherwise
        """
        crop_bounds = self.load_crop_bounds()

        print("Loading meshes...")
        for label, filepath in self.mesh_files:
            try:
                mesh = o3d.io.read_triangle_mesh(filepath)

                # If mesh has no triangles, treat it as a point cloud
                if len(mesh.triangles) == 0:
                    print(f"  {label}: No triangles found, loading as point cloud")
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = mesh.vertices
                    if mesh.has_vertex_colors() and not self.config.ignore_mesh_color:
                        pcd.colors = mesh.vertex_colors
                    else:
                        # Assign default gray color
                        pcd.paint_uniform_color([0.7, 0.7, 0.7])

                    if crop_bounds is not None:
                        pcd = pcd.crop(crop_bounds)
                        print("    Cropped point cloud to specified bounds")
                    self.meshes.append(pcd)
                else:
                    # Always recompute normals for better lighting
                    # Use consistent normal estimation for smooth shading
                    mesh.compute_vertex_normals()

                    # For enhanced lighting, also compute triangle normals
                    if self.enable_enhanced_lighting:
                        mesh.compute_triangle_normals()

                    # If no colors, assign default light gray
                    if not mesh.has_vertex_colors() or self.config.ignore_mesh_color:
                        mesh.paint_uniform_color([0.8, 0.8, 0.8])

                    if crop_bounds is not None:
                        mesh = mesh.crop(crop_bounds)
                        print("    Cropped mesh to specified bounds")
                    self.meshes.append(mesh)

                print(f"  Loaded: {label} from {filepath}")

            except Exception as e:
                print(f"  Error loading {label} from {filepath}: {e}")
                return False

        # Apply error coloring if enabled
        if self.config.enable_error_coloring:
            if not self.apply_error_coloring():
                print("Warning: Failed to apply error coloring, meshes will use default colors")

        return len(self.meshes) == len(self.mesh_files)

    def load_ground_truth(self) -> Optional[o3d.geometry.PointCloud]:
        """
        Load and sample ground truth mesh/point cloud.

        Returns:
            Point cloud sampled from ground truth, or None if failed
        """
        if self.config.ground_truth_mesh_idx >= 0:
            print("Using ground truth from specified mesh index")
            gt_mesh = self.meshes[self.config.ground_truth_mesh_idx]
        else:
            if not self.config.ground_truth_file:
                print("Error: ground_truth_file not specified")
                return None

            print(f"\nLoading ground truth from: {self.config.ground_truth_file}")

            # Try loading as mesh first
            gt_mesh = o3d.io.read_triangle_mesh(self.config.ground_truth_file)

        if len(gt_mesh.triangles) > 0:
            # It's a mesh - sample points from it
            print(f"  Ground truth is a mesh with {len(gt_mesh.triangles)} triangles")
            print(f"  Sampling {self.config.gt_sample_points} points...")
            gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=self.config.gt_sample_points)
        else:
            # It's a point cloud
            print(f"  Ground truth is a point cloud with {len(gt_mesh.vertices)} points")
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = gt_mesh.vertices

            # If too many points, downsample
            if len(gt_pcd.points) > self.config.gt_sample_points:
                print(f"  Downsampling to {self.config.gt_sample_points} points...")
                gt_pcd = gt_pcd.farthest_point_down_sample(self.config.gt_sample_points)

        print(f"  Ground truth point cloud ready: {len(gt_pcd.points)} points")
        return gt_pcd

    def compute_vertex_errors(self, mesh: o3d.geometry.TriangleMesh, gt_pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Compute distance from each mesh vertex to nearest ground truth point.

        Args:
            mesh: Mesh to compute errors for
            gt_pcd: Ground truth point cloud

        Returns:
            Array of distances (errors) for each vertex
        """
        # Get mesh vertices
        vertices = np.asarray(mesh.vertices)
        gt_points = np.asarray(gt_pcd.points)

        print(f"    Computing errors for {len(vertices)} vertices...")

        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(gt_points)

        # Query nearest neighbor for each vertex
        distances, _ = tree.query(vertices, k=1)

        return distances

    def jet_colormap(self, value: np.ndarray) -> np.ndarray:
        """
        Jet colormap: blue -> cyan -> green -> yellow -> red

        Args:
            value: Normalized value in [0, 1]

        Returns:
            RGB color as numpy array [R, G, B] in range [0, 1]
        """
        # Clamp value to [0, 1]
        value = np.clip(value, 0.0, 1.0)
        value = (value * 255).astype(np.uint8)
        color = cv2.applyColorMap(value, cv2.COLORMAP_JET)
        return color[:, 0, ::-1] / 255.0  # Convert BGR to RGB and normalize

    def apply_error_coloring(self) -> bool:
        """
        Apply error-based coloring to all meshes using jet colormap.

        Returns:
            True if successful, False otherwise
        """

        print("\n" + "=" * 70)
        print("APPLYING ERROR-BASED COLORING")
        print("=" * 70)

        # Load ground truth
        gt_pcd = self.load_ground_truth()
        if gt_pcd is None:
            return False

        # Process each mesh
        for idx, (mesh, (label, _)) in enumerate(zip(self.meshes, self.mesh_files)):
            if idx == self.config.ground_truth_mesh_idx:
                print(f"\n  Skipping {label}: Ground truth mesh")
                continue
            # Only process triangle meshes (skip point clouds)
            if not isinstance(mesh, o3d.geometry.TriangleMesh):
                print(f"\n  Skipping {label}: Not a triangle mesh")
                continue

            if len(mesh.triangles) == 0:
                print(f"\n  Skipping {label}: No triangles")
                continue

            print(f"\n  Processing {label}...")

            # Compute errors for each vertex
            errors = self.compute_vertex_errors(mesh, gt_pcd)

            # Statistics
            min_error = np.min(errors)
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)

            print(f"    Error statistics:")
            print(f"      Min:    {min_error:.6f} m")
            print(f"      Max:    {max_error:.6f} m")
            print(f"      Mean:   {mean_error:.6f} m")
            print(f"      Median: {median_error:.6f} m")
            print(f"      Std:    {std_error:.6f} m")

            # Normalize errors to [0, 1] based on error_min and error_max
            error_range = self.config.error_max - self.config.error_min
            if error_range <= 0:
                print(f"    Warning: Invalid error range [{self.config.error_min}, {self.config.error_max}]")
                continue  # Skip coloring
            else:
                normalized_errors = (errors - self.config.error_min) / error_range
                normalized_errors = np.clip(normalized_errors, 0.0, 1.0)

            # Apply jet colormap to each vertex
            colors = self.jet_colormap(normalized_errors)

            # Set vertex colors
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

            print(f"    ✓ Applied jet colormap coloring")
            print(f"      Color range: [{self.config.error_min}, {self.config.error_max}] m")

        if self.config.show_error_colorbar:
            self.print_error_colorbar()

        print("\n" + "=" * 70)
        print("ERROR COLORING COMPLETE")
        print("=" * 70 + "\n")

        return True

    def print_error_colorbar(self):
        """
        Print a text-based colorbar legend to console.
        """
        print("\n    Error Colorbar (Jet):")
        print("    " + "-" * 50)

        # Print colorbar
        bar = "    "
        for i in range(50):
            val = i / 49.0
            bar += "█"
        print(bar)

        # Print scale
        error_range = self.config.error_max - self.config.error_min
        print(f"    {self.config.error_min:.4f}m" + " " * 35 + f"{self.config.error_max:.4f}m")
        print(f"    (Min Error)" + " " * 35 + "(Max Error)")

        # Print key points
        print("\n    Color Guide:")
        print(f"      Blue   → Low error   (≤ {self.config.error_min + 0.25 * error_range:.4f}m)")
        print(f"      Green  → Medium error (≈ {self.config.error_min + 0.5 * error_range:.4f}m)")
        print(f"      Red    → High error  (≥ {self.config.error_min + 0.75 * error_range:.4f}m)")
        print("    " + "-" * 50)

    def load_camera_config(self) -> bool:
        """
        Load camera configuration from JSON file.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self.camera_config_file is None:
            return False

        try:
            config_path = pathlib.Path(self.camera_config_file)
            if not config_path.exists():
                print(f"Warning: Camera config file not found: {self.camera_config_file}")
                return False

            with open(config_path, "r") as f:
                self.camera_config = json.load(f)

            # Validate it's a ViewTrajectory format
            if self.camera_config.get("class_name") != "ViewTrajectory":
                print(f"Warning: Invalid camera config format (not ViewTrajectory)")
                return False

            if not self.camera_config.get("trajectory") or len(self.camera_config["trajectory"]) == 0:
                print(f"Warning: No camera trajectory found in config file")
                return False

            print(f"Loaded camera configuration from: {self.camera_config_file}")
            return True

        except Exception as e:
            print(f"Error loading camera config: {e}")
            return False

    def apply_camera_config(self, view_control):
        """
        Apply loaded camera configuration to a view control.

        Args:
            view_control: Open3D view control object
        """
        if self.camera_config is None:
            return

        try:
            # Get the first trajectory point
            trajectory = self.camera_config["trajectory"][0]

            # Extract camera parameters
            front = trajectory.get("front", [0, 0, -1])
            lookat = trajectory.get("lookat", [0, 0, 0])
            up = trajectory.get("up", [0, -1, 0])
            zoom = trajectory.get("zoom", 0.7)

            # Apply to view control
            view_control.set_front(front)
            view_control.set_lookat(lookat)
            view_control.set_up(up)
            view_control.set_zoom(zoom)

            print(f"  Applied camera config: front={front}, zoom={zoom}")

        except Exception as e:
            print(f"Error applying camera config: {e}")

    def create_visualizers(self):
        """
        Create visualization windows for each mesh.

        Window Layout:
        - Uses n_rows and n_cols from config to determine grid layout
        - If both n_rows and n_cols > 0: Uses specified grid (may leave empty cells)
        - If only n_rows > 0: Arranges in n_rows rows (calculates columns)
        - If only n_cols > 0: Arranges in n_cols columns (calculates rows)
        - If both = 0: Auto-calculates square-ish grid layout
        - Windows arranged in row-major order (left to right, top to bottom)

        Rendering Configuration:
        - Backface culling: Enabled for better performance and clarity
        - Lighting: Enhanced mode with Phong shading for depth perception
        - Background: Customizable via background_color config (default: dark gray [0.05, 0.05, 0.05])
        - Normals: Computed for all meshes to ensure proper lighting
        - Shading: Uses vertex normals for smooth surface appearance with shadows

        The combination of proper normals, Phong shading, and appropriate background color
        creates visible shadows and highlights that reveal surface structure.
        """
        print("\nCreating visualization windows...")

        # Calculate window positions (arrange in a grid)
        num_meshes = len(self.meshes)

        # Determine grid layout based on n_rows and n_cols config
        if self.config.n_rows > 0 and self.config.n_cols > 0:
            # Both specified - use as-is
            rows = self.config.n_rows
            cols = self.config.n_cols
            if rows * cols < num_meshes:
                print(f"Warning: Grid {rows}×{cols} = {rows * cols} cells, but {num_meshes} meshes to display")
                print("Some meshes will not be displayed. Increase n_rows or n_cols.")
        elif self.config.n_rows > 0:
            # Only rows specified - calculate columns
            rows = self.config.n_rows
            cols = int(np.ceil(num_meshes / rows))
        elif self.config.n_cols > 0:
            # Only columns specified - calculate rows
            cols = self.config.n_cols
            rows = int(np.ceil(num_meshes / cols))
        else:
            # Auto-calculate square-ish grid layout
            cols = int(np.ceil(np.sqrt(num_meshes)))
            rows = int(np.ceil(num_meshes / cols))

        offset_x = 50
        offset_y = 50

        print(f"  Layout: {rows} rows × {cols} columns ({num_meshes} meshes)")

        for idx, (mesh, (label, _)) in enumerate(zip(self.meshes, self.mesh_files)):
            # Calculate position in grid (row-major order)
            row = idx // cols
            col = idx % cols

            # Create visualizer
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(
                window_name=f"{self.window_name} - {label}",
                width=self.window_width,
                height=self.window_height,
                left=offset_x + col * (self.window_width + 10),
                top=offset_y + row * (self.window_height + 40),
            )

            # Add geometry
            vis.add_geometry(mesh)

            # Register key callbacks for this visualizer
            # Save screenshots callback
            def save_screenshots_callback(vis_obj):
                current_time = time.time()
                # Debounce to prevent multiple saves
                if not hasattr(self, "_last_screenshot_time"):
                    self._last_screenshot_time = 0
                if current_time - self._last_screenshot_time > 1.0:
                    self._last_screenshot_time = current_time
                    # Save screenshots with timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    self.save_screenshots(timestamp)
                return False

            vis.register_key_callback(ord("P"), save_screenshots_callback)
            vis.register_key_callback(ord("p"), save_screenshots_callback)

            # Get render option and configure advanced lighting/rendering
            render_option = vis.get_render_option()

            # Enable backface culling for better performance and clarity
            render_option.mesh_show_back_face = not self.config.backface_culling

            if self.enable_enhanced_lighting:
                # Enhanced lighting mode for better surface structure visualization
                render_option.light_on = True

                # Shading options available in Open3D:
                # - Default: Phong shading with vertex normals (smooth, shows depth well)
                # - Color: Uses vertex colors only (flat, no lighting-based shadows)
                # - Normal: Visualizes normals as RGB colors (debug mode)
                # We use Default for best depth perception with shadows
                render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Default

                # Enable proper mesh coloring with lighting effects
                if isinstance(mesh, o3d.geometry.TriangleMesh):
                    # Use Color option to display vertex colors (e.g., from error coloring)
                    # If mesh has vertex colors, they will be shown with Phong lighting
                    if mesh.has_vertex_colors():
                        render_option.mesh_color_option = o3d.visualization.MeshColorOption.Color
                    else:
                        render_option.mesh_color_option = o3d.visualization.MeshColorOption.Default
                else:
                    if mesh.has_colors():
                        render_option.point_color_option = o3d.visualization.PointColorOption.Color
                    else:
                        render_option.point_color_option = o3d.visualization.PointColorOption.Default

                # Adjust point cloud rendering (if applicable)
                render_option.point_size = 3.0
                render_option.point_show_normal = False

            else:
                # Standard lighting mode
                render_option.light_on = True
                render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
                render_option.point_size = 2.0

                if isinstance(mesh, o3d.geometry.TriangleMesh):
                    # Use Color option to display vertex colors
                    if mesh.has_vertex_colors():
                        render_option.mesh_color_option = o3d.visualization.MeshColorOption.Color
                    else:
                        render_option.mesh_color_option = o3d.visualization.MeshColorOption.Default
                else:
                    if mesh.has_colors():
                        render_option.point_color_option = o3d.visualization.PointColorOption.Color
                    else:
                        render_option.point_color_option = o3d.visualization.PointColorOption.Default

            # Set background color from config (works for both lighting modes)
            render_option.background_color = self.parse_background_color(self.config.background_color)

            # Get view control
            view_control = vis.get_view_control()

            # Store visualizer and view control
            self.visualizers.append(vis)
            self.view_controls.append(view_control)

            print(f"  Created window for: {label}")

        # Set the first view control as reference
        if len(self.view_controls) > 0:
            self.reference_view_control = self.view_controls[0]

    def get_camera_parameters(self, view_control) -> dict:
        """
        Extract camera parameters from a view control.

        Args:
            view_control: Open3D view control object

        Returns:
            Dictionary containing camera parameters
        """
        params = view_control.convert_to_pinhole_camera_parameters()
        return {
            "extrinsic": params.extrinsic.copy(),
            "intrinsic": params.intrinsic.intrinsic_matrix.copy(),
            "width": params.intrinsic.width,
            "height": params.intrinsic.height,
        }

    def set_camera_parameters(self, view_control, params: dict):
        """
        Apply camera parameters to a view control.

        Args:
            view_control: Open3D view control object
            params: Dictionary containing camera parameters
        """
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.extrinsic = params["extrinsic"]

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            params["width"],
            params["height"],
            params["intrinsic"][0, 0],
            params["intrinsic"][1, 1],
            params["intrinsic"][0, 2],
            params["intrinsic"][1, 2],
        )
        camera_params.intrinsic = intrinsic

        view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    def synchronize_cameras(self):
        """
        Synchronize all camera views to match the reference view.
        """
        if not self.sync_enabled or self.reference_view_control is None:
            return

        with self.sync_lock:
            try:
                # Get current camera parameters from reference
                current_params = self.get_camera_parameters(self.reference_view_control)

                # Check if camera has moved
                if self.last_camera_params is not None:
                    # Compare extrinsic matrices to detect changes
                    if np.allclose(current_params["extrinsic"], self.last_camera_params["extrinsic"], rtol=1e-6):
                        return  # No change detected

                # Update camera for all other views
                for view_control in self.view_controls[1:]:
                    self.set_camera_parameters(view_control, current_params)

                # Store current parameters
                self.last_camera_params = current_params

            except Exception as e:
                print(f"Error synchronizing cameras: {e}")

    def update_visualizers(self) -> bool:
        """
        Update all visualizers and handle events.

        Returns:
            True if all windows are still open, False otherwise
        """
        all_open = True

        for vis in self.visualizers:
            if not vis.poll_events():
                all_open = False
            vis.update_renderer()

        # Synchronize cameras after updates
        self.synchronize_cameras()

        return all_open

    def save_screenshots(self, timestamp: str = None):
        """
        Save screenshots of all visualizer windows.

        Args:
            timestamp: Optional timestamp string to use in filenames.
                      If None, current time will be used.
        """
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create output directory if it doesn't exist
        output_dir = pathlib.Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📸 Saving screenshots to: {output_dir}/")

        saved_count = 0
        for idx, (vis, (label, _)) in enumerate(zip(self.visualizers, self.mesh_files)):
            try:
                # Create filename with label and timestamp
                # Sanitize label for filename (replace spaces and special chars)
                safe_label = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
                filename = f"{safe_label}.png"

                # Full path with output directory
                filepath = output_dir / filename

                # Capture and save screenshot
                vis.capture_screen_image(str(filepath), do_render=True)

                print(f"  ✓ Saved: {filepath}")
                saved_count += 1

            except Exception as e:
                print(f"  ✗ Error saving screenshot for {label}: {e}")

        if saved_count > 0:
            print(f"\n✓ Successfully saved {saved_count}/{len(self.visualizers)} screenshots")
        else:
            print("\n✗ Failed to save any screenshots")

    def print_controls(self):
        """
        Print control instructions to console.
        """
        print("\n" + "=" * 60)
        print("SYNCHRONIZED MESH VIEWER CONTROLS")
        print("=" * 60)
        print("Mouse Controls:")
        print("  - Left button + drag:   Rotate camera")
        print("  - Right button + drag:  Pan camera")
        print("  - Scroll wheel:         Zoom in/out")
        print("  - Middle button + drag: Pan camera (alternative)")
        print("\nKeyboard Controls:")
        print("  - R: Reset camera view")
        print("  - S: Toggle camera synchronization on/off")
        print("  - Ctrl+C: Save current camera configuration to file")
        print("  - P: Save screenshots of all windows")
        print("  - L: Toggle lighting on/off")
        print("  - H: Print this help message")
        print("  - Q/ESC: Quit application")
        print("\nRendering Features:")
        print(f"  - Backface culling: {self.config.backface_culling}")
        print("  - Enhanced lighting: " + ("Enabled" if self.enable_enhanced_lighting else "Disabled"))
        print("  - Phong shading with vertex normals (shows shadows)")
        print(f"  - Background color: {self.config.background_color}")
        print("  - Triangle & vertex normals computed for all meshes")
        if self.config.enable_error_coloring:
            print("\nError Coloring:")
            print(f"  - Enabled: Yes")
            print(f"  - Ground truth: {self.config.ground_truth_file}")
            print(f"  - Error range: [{self.config.error_min}, {self.config.error_max}] m")
            print(f"  - Colormap: Jet (Blue → Cyan → Green → Yellow → Red)")
        if self.camera_config_file:
            print("\nCamera Configuration:")
            print(f"  - Loaded from: {self.camera_config_file}")
        print("\nNote: All camera movements in the first window will be")
        print("      synchronized across all other windows.")
        print("=" * 60 + "\n")

    def run(self):
        """
        Main visualization loop.
        """
        if not self.load_meshes():
            print("Failed to load all meshes. Exiting.")
            return

        # Load camera configuration if specified
        has_camera_config = self.load_camera_config()

        self.create_visualizers()
        self.print_controls()

        # Set initial camera view and ensure proper rendering for all visualizers
        print("Setting initial camera views and rendering options...")
        for idx, (vis, view_control) in enumerate(zip(self.visualizers, self.view_controls)):
            # Set camera view (either from config or default)
            if has_camera_config:
                self.apply_camera_config(view_control)
            else:
                # Default camera view
                view_control.set_front([0, 0, -1])
                view_control.set_up([0, -1, 0])
                view_control.set_zoom(0.7)

            # Force update geometry to ensure normals are properly computed
            # This helps with lighting calculations
            vis.update_geometry(self.meshes[idx])
            vis.update_renderer()

        # Synchronize initial view
        self.synchronize_cameras()

        print("\nStarting visualization loop...")
        print("Move the camera in the first window to see synchronized views.")
        print("Press 'P' to save screenshots.\n")

        # Main loop
        try:
            while self.is_running:
                # Update all visualizers
                if not self.update_visualizers():
                    print("\nOne or more windows were closed. Exiting...")
                    break

                # Small sleep to prevent CPU spinning
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Exiting...")

        finally:
            self.cleanup()

    def cleanup(self):
        """
        Clean up and destroy all visualizers.
        """
        print("Cleaning up visualizers...")
        for vis in self.visualizers:
            try:
                vis.destroy_window()
            except:
                pass
        print("Done!")


def main():
    """
    Main entry point for the mesh comparison visualization tool.
    """
    parser = ViewerConfig.get_argparser()
    cfg: ViewerConfig = parser.parse_args()

    assert cfg.mesh_files is not None and len(cfg.mesh_files) > 0, "At least one mesh file must be specified"

    # Camera configuration file handling:
    # Priority:
    # 1. Command-line argument (--camera-config)
    # 2. Auto-detect in script directory (view_status_o3d.json)
    # 3. None (use default camera)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if cfg.camera_config_file:
        # Use command-line specified config
        if not os.path.exists(cfg.camera_config_file):
            print(f"Warning: Specified camera config file not found: {cfg.camera_config_file}")
            print("Using default camera settings.\n")
            cfg.camera_config_file = None
    else:
        # Try to auto-detect in script directory
        camera_config_file = os.path.join(script_dir, "view_status_o3d.json")
        if os.path.exists(camera_config_file):
            cfg.camera_config_file = camera_config_file
        else:
            print("Camera config file not found at: {}".format(camera_config_file))
            print("Using default camera settings. You can save your preferred view by pressing 'C'.\n")

    # Create and run the viewer
    viewer = SynchronizedMeshViewer(cfg)
    viewer.run()


if __name__ == "__main__":
    main()

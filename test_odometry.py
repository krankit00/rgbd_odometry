#!/usr/bin/env python3

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import cv2
from utils.odometry_methods import OdometryMethods

def load_sample_rgbd_data():
    """Load sample RGBD data from Open3D."""
    print("Downloading sample RGBD dataset...")
    redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
    
    # Get paths to color and depth images
    color_paths = []
    depth_paths = []
    
    for i in range(5):
        color_paths.append(os.path.join(redwood_rgbd.color_paths[i]))
        depth_paths.append(os.path.join(redwood_rgbd.depth_paths[i]))
    
    return color_paths, depth_paths

class TrajectoryVisualizer:
    def __init__(self):
        """Initialize the trajectory visualizer."""
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []
        self.setup_plot()
        plt.show()
        
    def setup_plot(self):
        """Setup the 3D plot."""
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Camera Trajectory')
        self.ax.view_init(elev=20, azim=45)
        
    def update_plot(self):
        """Update the plot with current trajectory."""
        self.ax.cla()
        self.setup_plot()
        
        if len(self.trajectory_x) > 0:
            # Plot trajectory line
            self.ax.plot3D(self.trajectory_x, self.trajectory_y, self.trajectory_z, 'b-', label='Path')
            
            # Plot current position
            self.ax.scatter(self.trajectory_x[-1], self.trajectory_y[-1], self.trajectory_z[-1], 
                          c='red', marker='o', s=100, label='Current Position')
            
            # Plot start position
            self.ax.scatter(self.trajectory_x[0], self.trajectory_y[0], self.trajectory_z[0], 
                          c='green', marker='o', s=100, label='Start')
            
            # Adjust plot limits with some padding
            max_range = max([
                np.max(np.abs(self.trajectory_x)),
                np.max(np.abs(self.trajectory_y)),
                np.max(np.abs(self.trajectory_z))
            ]) * 1.1
            
            if max_range > 0:
                self.ax.set_xlim([-max_range, max_range])
                self.ax.set_ylim([-max_range, max_range])
                self.ax.set_zlim([-max_range, max_range])
            self.ax.legend()
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def add_point(self, transform):
        """Add a new point to the trajectory."""
        position = transform[:3, 3]  # Extract translation part
        self.trajectory_x.append(position[0])
        self.trajectory_y.append(position[1])
        self.trajectory_z.append(position[2])
        self.update_plot()

class FrameVisualizer:
    def __init__(self):
        """Initialize frame-by-frame visualizer."""
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.canvas.manager.set_window_title('Frame-by-Frame Odometry')
        
        # Flatten axes for easier access
        self.axes = self.axes.flatten()
        
        # Set titles
        self.axes[0].set_title('Source RGB')
        self.axes[1].set_title('Source Depth')
        self.axes[2].set_title('Target RGB')
        self.axes[3].set_title('Target Depth')
        
        plt.tight_layout()
        plt.show()
    
    def update(self, source_color, source_depth, target_color, target_depth, trans=None, intrinsics=None):
        """Update visualization with new frames."""
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        # Show RGB images
        self.axes[0].imshow(source_color)
        self.axes[2].imshow(target_color)
        
        # Show depth images with consistent colormap
        depth_vmin = min(np.min(source_depth[source_depth > 0]), np.min(target_depth[target_depth > 0]))
        depth_vmax = max(np.max(source_depth), np.max(target_depth))
        
        self.axes[1].imshow(source_depth, cmap='jet', vmin=depth_vmin, vmax=depth_vmax)
        self.axes[3].imshow(target_depth, cmap='jet', vmin=depth_vmin, vmax=depth_vmax)
        
        # Get center point depth values
        h, w = source_depth.shape
        center_y, center_x = h // 2, w // 2
        source_center_depth = source_depth[center_y, center_x]
        target_center_depth = target_depth[center_y, center_x]
        
        # Plot center points
        self.axes[1].plot(center_x, center_y, 'r+', markersize=10)
        self.axes[3].plot(center_x, center_y, 'r+', markersize=10)
        
        # Add depth info to titles
        self.axes[1].set_title(f'Source Depth (center: {source_center_depth:.3f}m)')
        self.axes[3].set_title(f'Target Depth (center: {target_center_depth:.3f}m)')
        
        # Add transformation info if available
        info_text = ""
        if trans is not None:
            translation = trans[:3, 3]
            rot_mat = trans[:3, :3]
            euler = np.degrees(np.array([
                np.arctan2(rot_mat[2, 1], rot_mat[2, 2]),
                np.arctan2(-rot_mat[2, 0], np.sqrt(rot_mat[2, 1]**2 + rot_mat[2, 2]**2)),
                np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
            ]))
            info_text += f'Translation (m): ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})\n'
            info_text += f'Rotation (deg): ({euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f})\n'
        
        # Add intrinsics info if available
        if intrinsics is not None:
            if isinstance(intrinsics, o3d.camera.PinholeCameraIntrinsic):
                matrix = np.asarray(intrinsics.intrinsic_matrix)
                info_text += f'\nIntrinsics Matrix:\n'
                info_text += f'fx: {matrix[0,0]:.1f}, fy: {matrix[1,1]:.1f}\n'
                info_text += f'cx: {matrix[0,2]:.1f}, cy: {matrix[1,2]:.1f}\n'
                info_text += f'width: {intrinsics.width}, height: {intrinsics.height}'
        
        if info_text:
            self.fig.suptitle(info_text, fontsize=10)
        
        # Remove axes
        for ax in self.axes:
            ax.axis('off')
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main():
    # Load sample data
    color_paths, depth_paths = load_sample_rgbd_data()
    
    # Load configuration
    with open('config/odometry_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize visualizers
    trajectory_vis = TrajectoryVisualizer()
    frame_vis = FrameVisualizer()
    
    # Get camera intrinsics (from Open3D sample data)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    
    print("\nCamera Intrinsics:")
    print(np.asarray(intrinsic.intrinsic_matrix))
    
    # Initialize odometry
    odometry = OdometryMethods(config)
    
    # Initialize trajectory
    global_transform = np.identity(4)
    trajectory = [global_transform.copy()]
    trajectory_vis.add_point(global_transform)
    
    # Process frames
    for i in tqdm(range(len(color_paths) - 1), desc="Computing odometry"):
        # Read source and target images
        source_color = o3d.io.read_image(color_paths[i])
        source_depth = o3d.io.read_image(depth_paths[i])
        target_color = o3d.io.read_image(color_paths[i + 1])
        target_depth = o3d.io.read_image(depth_paths[i + 1])
        
        # Print raw depth statistics
        source_depth_np_raw = np.asarray(source_depth)
        print(f"\nRaw depth statistics for frame {i}:")
        print(f"Min depth: {np.min(source_depth_np_raw[source_depth_np_raw > 0])}")
        print(f"Max depth: {np.max(source_depth_np_raw)}")
        print(f"Mean depth: {np.mean(source_depth_np_raw[source_depth_np_raw > 0])}")
        
        # Convert to numpy arrays for visualization
        source_color_np = np.asarray(source_color)
        source_depth_np = np.asarray(source_depth).astype(float) / 1000.0  # Convert to meters
        target_color_np = np.asarray(target_color)
        target_depth_np = np.asarray(target_depth).astype(float) / 1000.0  # Convert to meters
        
        # Print converted depth statistics
        print(f"\nConverted depth statistics (meters) for frame {i}:")
        print(f"Min depth: {np.min(source_depth_np[source_depth_np > 0]):.3f}m")
        print(f"Max depth: {np.max(source_depth_np):.3f}m")
        print(f"Mean depth: {np.mean(source_depth_np[source_depth_np > 0]):.3f}m")
        
        # Create RGBD images with proper scaling
        source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            source_color, source_depth,
            depth_scale=1000.0,  # Input depth is in millimeters
            depth_trunc=config['camera']['depth_trunc'],
            convert_rgb_to_intensity=True)
        
        target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            target_color, target_depth,
            depth_scale=1000.0,  # Input depth is in millimeters
            depth_trunc=config['camera']['depth_trunc'],
            convert_rgb_to_intensity=True)
        
        # Print RGBD image depth statistics
        print(f"\nRGBD depth statistics for frame {i}:")
        source_depth_o3d = np.asarray(source_rgbd.depth)
        print(f"Min RGBD depth: {np.min(source_depth_o3d[source_depth_o3d > 0]):.3f}m")
        print(f"Max RGBD depth: {np.max(source_depth_o3d):.3f}m")
        print(f"Mean RGBD depth: {np.mean(source_depth_o3d[source_depth_o3d > 0]):.3f}m")
        
        # Compute odometry
        success, trans, info = odometry.compute_odometry(
            source_rgbd, target_rgbd,
            intrinsic
        )
        
        # Update visualization
        frame_vis.update(
            source_color_np, source_depth_np,
            target_color_np, target_depth_np,
            trans if success else None,
            intrinsic
        )
        
        if success:
            # Scale the translation part of the transformation (if needed)
            # trans[:3, 3] *= 1.0  # Uncomment and adjust if scaling is needed
            
            global_transform = np.dot(global_transform, trans)
            trajectory.append(global_transform.copy())
            trajectory_vis.add_point(global_transform)
            
            # Print odometry information
            translation = trans[:3, 3]
            print(f"\nFrame {i} -> {i+1} transformation:")
            print(f"Translation (m): {translation}")
            print(f"Full transformation matrix:")
            print(trans)
            if hasattr(info, 'fitness'):
                print(f"Fitness: {info.fitness:.4f}")
        else:
            print(f"\nFailed to compute odometry for frame {i} -> {i+1}")
        
        plt.pause(0.1)  # Small delay to allow visualization
    
    # Save trajectory
    trajectory = np.array(trajectory)
    np.save('sample_trajectory.npy', trajectory)
    print(f"\nSaved trajectory with {len(trajectory)} poses")
    
    # Print final trajectory statistics
    total_translation = np.linalg.norm(trajectory[-1][:3, 3] - trajectory[0][:3, 3])
    print(f"\nTotal trajectory length: {total_translation:.3f} meters")
    
    # Print average frame-to-frame translation
    frame_translations = []
    for i in range(1, len(trajectory)):
        trans = np.linalg.norm(trajectory[i][:3, 3] - trajectory[i-1][:3, 3])
        frame_translations.append(trans)
    print(f"Average frame-to-frame translation: {np.mean(frame_translations):.3f} meters")
    print(f"Min frame-to-frame translation: {np.min(frame_translations):.3f} meters")
    print(f"Max frame-to-frame translation: {np.max(frame_translations):.3f} meters")
    
    plt.show(block=True)

if __name__ == "__main__":
    main() 
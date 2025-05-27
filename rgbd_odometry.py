#!/usr/bin/env python3

import os
import cv2
import numpy as np
import open3d as o3d
import rospy
import cv_bridge
from cv_bridge import CvBridge
import pyrealsense2 as rs
from rospy.rostime import Time
from rosbag import Bag
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
import signal
import sys
import yaml
from utils.odometry_methods import OdometryMethods

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print('\nGracefully shutting down...')
    running = False
    plt.close('all')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class TrajectoryVisualizer:
    def __init__(self):
        """Initialize the trajectory visualizer."""
        plt.ion()  # Enable interactive mode
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
            max_range = np.max([
                np.max(np.abs(self.trajectory_x)),
                np.max(np.abs(self.trajectory_y)),
                np.max(np.abs(self.trajectory_z))
            ]) * 1.1
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
        self.axes[0].imshow(cv2.cvtColor(source_color, cv2.COLOR_BGR2RGB))
        self.axes[2].imshow(cv2.cvtColor(target_color, cv2.COLOR_BGR2RGB))
        
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
            euler = rotation_matrix_to_euler_angles(rot_mat)
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
            else:
                # For RealSense intrinsics
                info_text += f'\nIntrinsics:\n'
                info_text += f'fx: {intrinsics.fx:.1f}, fy: {intrinsics.fy:.1f}\n'
                info_text += f'cx: {intrinsics.ppx:.1f}, cy: {intrinsics.ppy:.1f}\n'
                info_text += f'width: {intrinsics.width}, height: {intrinsics.height}'
        
        if info_text:
            self.fig.suptitle(info_text, fontsize=10)
        
        # Update titles
        self.axes[0].set_title('Source RGB')
        self.axes[2].set_title('Target RGB')
        
        # Remove axes
        for ax in self.axes:
            ax.axis('off')
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to euler angles (in degrees)."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw]) * 180.0 / np.pi

class RGBDOdometryProcessor:
    def __init__(self, bag_path, config_path, output_dir="./data"):
        """Initialize the RGBD Odometry processor."""
        self.bag_path = bag_path
        self.output_dir = output_dir
        self.bridge = CvBridge()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directories
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.depth_dir = os.path.join(output_dir, "depth")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        
        # Initialize visualizers
        self.trajectory_vis = TrajectoryVisualizer()
        self.frame_vis = FrameVisualizer()
        
        # Initialize camera intrinsics
        self.init_camera_intrinsics()
        
        # Initialize odometry methods
        self.odometry = OdometryMethods(self.config)

    def print_intrinsics_comparison(self, realsense_intr):
        """Print detailed comparison of intrinsics."""
        print("\n=== Camera Intrinsics Comparison ===")
        print("RealSense Intrinsics:")
        print(f"Resolution: {realsense_intr.width}x{realsense_intr.height}")
        print(f"Focal Length: fx={realsense_intr.fx:.2f}, fy={realsense_intr.fy:.2f}")
        print(f"Principal Point: cx={realsense_intr.ppx:.2f}, cy={realsense_intr.ppy:.2f}")
        print(f"Distortion Model: {realsense_intr.model}")
        print(f"Distortion Coefficients: {realsense_intr.coeffs}")
        
        print("\nOpen3D Intrinsics Matrix:")
        print(self.intrinsic.intrinsic_matrix)
        
        # Verify matrix composition
        print("\nVerifying intrinsics matrix composition:")
        expected_matrix = np.array([
            [realsense_intr.fx, 0, realsense_intr.ppx],
            [0, realsense_intr.fy, realsense_intr.ppy],
            [0, 0, 1]
        ])
        print("Expected matrix:")
        print(expected_matrix)
        
        print("\nMatrix difference:")
        diff = self.intrinsic.intrinsic_matrix - expected_matrix
        print(diff)
        print(f"Max absolute difference: {np.abs(diff).max():.2e}")

    def init_camera_intrinsics(self):
        """Initialize RealSense camera intrinsics."""
        print("\nInitializing camera intrinsics...")
        pipeline = rs.pipeline()
        config = rs.config()
        
        try:
            # Enable all streams from the bag file
            rs.config.enable_device_from_file(config, self.bag_path, repeat_playback=False)
            profile = pipeline.start(config)
            
            # Get the first frameset
            for _ in range(30):  # Skip some frames to ensure stable streaming
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                    if frames:
                        break
                except Exception as e:
                    print(f"Warning: Failed to get frame: {e}")
            
            if not frames:
                raise Exception("Failed to get valid frames from the bag file")
            
            # Get stream profiles
            color_profile = frames.get_color_frame().get_profile().as_video_stream_profile()
            depth_profile = frames.get_depth_frame().get_profile().as_video_stream_profile()
            
            # Get intrinsics
            color_intrin = color_profile.get_intrinsics()
            depth_intrin = depth_profile.get_intrinsics()
            
            # Get extrinsics (color to depth)
            extrinsics = color_profile.get_extrinsics_to(depth_profile)
            
            pipeline.stop()
            
            # Create Open3D camera intrinsic object
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=color_intrin.width,
                height=color_intrin.height,
                fx=color_intrin.fx,
                fy=color_intrin.fy,
                cx=color_intrin.ppx,
                cy=color_intrin.ppy
            )
            
            # Store additional parameters
            self.color_intrin = color_intrin
            self.depth_intrin = depth_intrin
            self.color_to_depth_extrinsics = extrinsics
            
            # Print detailed comparison
            self.print_intrinsics_comparison(color_intrin)
            
        except Exception as e:
            print(f"Warning: Could not get intrinsics from bag file: {e}")
            print("Using default D435 intrinsics for 640x480 resolution")
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=640, height=480,
                fx=386.526, fy=386.526,
                cx=320.0, cy=240.0
            )

    def extract_images_from_bag(self):
        """Extract RGB and depth images from the rosbag."""
        print("Extracting images from rosbag...")
        
        with Bag(self.bag_path, 'r') as bag:
            rgb_topic = "/camera/color/image_raw"
            depth_topic = "/camera/depth/image_rect_raw"
            
            total_msgs = bag.get_message_count(topic_filters=[rgb_topic, depth_topic])
            
            rgb_msgs = []
            depth_msgs = []
            
            with tqdm(total=total_msgs, desc="Reading messages") as pbar:
                for topic, msg, t in bag.read_messages(topics=[rgb_topic, depth_topic]):
                    if not running:
                        break
                    if topic == rgb_topic:
                        rgb_msgs.append((t.to_nsec(), msg))
                    elif topic == depth_topic:
                        depth_msgs.append((t.to_nsec(), msg))
                    pbar.update(1)
            
            rgb_msgs.sort(key=lambda x: x[0])
            depth_msgs.sort(key=lambda x: x[0])
            
            print("Saving synchronized image pairs...")
            for i, (rgb_time, rgb_msg) in enumerate(tqdm(rgb_msgs, desc="Saving images")):
                if not running:
                    break
                closest_depth = min(depth_msgs, key=lambda x: abs(x[0] - rgb_time))
                
                rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
                depth_img = self.bridge.imgmsg_to_cv2(closest_depth[1], desired_encoding='16UC1')
                
                cv2.imwrite(os.path.join(self.rgb_dir, f"{i:06d}.png"), rgb_img)
                cv2.imwrite(os.path.join(self.depth_dir, f"{i:06d}.png"), depth_img)
        
        print(f"Extracted {len(rgb_msgs)} image pairs")
        return len(rgb_msgs)

    def compute_odometry(self):
        """Compute RGBD odometry using Open3D."""
        print("\nComputing RGBD odometry...")
        print(f"Using method: {self.config['odometry']['method']}")
        
        # Get intrinsics matrix
        intrinsic_matrix = np.asarray(self.intrinsic.intrinsic_matrix)
        print("\nCamera Intrinsics:")
        print(f"Width: {self.intrinsic.width}, Height: {self.intrinsic.height}")
        print(f"Intrinsic Matrix:")
        print(intrinsic_matrix)
        print(f"fx: {intrinsic_matrix[0,0]:.2f}, fy: {intrinsic_matrix[1,1]:.2f}")
        print(f"cx: {intrinsic_matrix[0,2]:.2f}, cy: {intrinsic_matrix[1,2]:.2f}")
        
        rgb_files = sorted(os.listdir(self.rgb_dir))
        depth_files = sorted(os.listdir(self.depth_dir))
        
        global_transform = np.identity(4)
        trajectory = [global_transform.copy()]
        
        # Add initial position to visualizer
        self.trajectory_vis.add_point(global_transform)
        
        # Performance tracking
        total_time = 0
        successful_frames = 0
        frame_translations = []
        
        for i in tqdm(range(len(rgb_files) - 1), desc="Computing odometry"):
            if not running:
                break
                
            start_time = time.time()
            
            # Read source and target images
            source_color = cv2.imread(os.path.join(self.rgb_dir, rgb_files[i]))
            source_depth = cv2.imread(os.path.join(self.depth_dir, depth_files[i]), cv2.IMREAD_ANYDEPTH)
            target_color = cv2.imread(os.path.join(self.rgb_dir, rgb_files[i + 1]))
            target_depth = cv2.imread(os.path.join(self.depth_dir, depth_files[i + 1]), cv2.IMREAD_ANYDEPTH)
            
            if source_depth is None or target_depth is None:
                print(f"\nWarning: Failed to read depth images for frame {i}")
                continue

            # Print raw depth statistics
            print(f"\nRaw depth statistics for frame {i}:")
            valid_raw_depths = source_depth[source_depth > 0]
            if valid_raw_depths.size > 0:
                print(f"Min depth: {np.min(valid_raw_depths)} mm")
                print(f"Max depth: {np.max(valid_raw_depths)} mm")
                print(f"Mean depth: {np.mean(valid_raw_depths)} mm")
                print(f"Number of valid depth points: {valid_raw_depths.size}")
                print(f"Percentage of valid depth points: {(valid_raw_depths.size / source_depth.size * 100):.1f}%")
            else:
                print("Warning: No valid raw depth points found!")
                print(f"Total points: {source_depth.size}")
                print(f"Depth value range: [{np.min(source_depth)}, {np.max(source_depth)}]")
            
            # Convert depth values from millimeters (uint16) to meters (float32)
            depth_scale = self.config['camera']['depth_scale']
            source_depth_m = source_depth.astype(np.float32) / depth_scale
            target_depth_m = target_depth.astype(np.float32) / depth_scale
            
            # Print converted depth statistics
            print(f"\nConverted depth statistics (meters) for frame {i}:")
            valid_converted_depths = source_depth_m[source_depth_m > 0]
            if valid_converted_depths.size > 0:
                print(f"Min depth: {np.min(valid_converted_depths):.3f}m")
                print(f"Max depth: {np.max(valid_converted_depths):.3f}m")
                print(f"Mean depth: {np.mean(valid_converted_depths):.3f}m")
                print(f"Number of valid depth points: {valid_converted_depths.size}")
                print(f"Percentage of valid depth points: {(valid_converted_depths.size / source_depth_m.size * 100):.1f}%")
            else:
                print("Warning: No valid converted depth points found!")
                print(f"Total points: {source_depth_m.size}")
                print(f"Depth value range: [{np.min(source_depth_m):.3f}, {np.max(source_depth_m):.3f}]")
            
            # Print center point depth for debugging
            h, w = source_depth_m.shape
            center_y, center_x = h // 2, w // 2
            print(f"\nCenter point depth values:")
            print(f"Source center (raw): {source_depth[center_y, center_x]} mm")
            print(f"Target center (raw): {target_depth[center_y, center_x]} mm")
            print(f"Source center (meters): {source_depth_m[center_y, center_x]:.3f}m")
            print(f"Target center (meters): {target_depth_m[center_y, center_x]:.3f}m")
            
            # Create Open3D images
            source_color_o3d = o3d.geometry.Image(cv2.cvtColor(source_color, cv2.COLOR_BGR2RGB))
            target_color_o3d = o3d.geometry.Image(cv2.cvtColor(target_color, cv2.COLOR_BGR2RGB))
            source_depth_o3d = o3d.geometry.Image(source_depth_m)
            target_depth_o3d = o3d.geometry.Image(target_depth_m)
            
            # Create RGBD images
            source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                source_color_o3d, source_depth_o3d,
                depth_scale=1.0,  # Already in meters
                depth_trunc=self.config['camera']['depth_trunc'],
                convert_rgb_to_intensity=True)
            
            target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                target_color_o3d, target_depth_o3d,
                depth_scale=1.0,  # Already in meters
                depth_trunc=self.config['camera']['depth_trunc'],
                convert_rgb_to_intensity=True)
            
            # Print RGBD image depth statistics
            print(f"\nRGBD depth statistics for frame {i}:")
            source_depth_o3d_np = np.asarray(source_rgbd.depth)
            valid_depths = source_depth_o3d_np[source_depth_o3d_np > 0]
            
            if valid_depths.size > 0:
                print(f"Min RGBD depth: {np.min(valid_depths):.3f}m")
                print(f"Max RGBD depth: {np.max(valid_depths):.3f}m")
                print(f"Mean RGBD depth: {np.mean(valid_depths):.3f}m")
                print(f"Number of valid depth points: {valid_depths.size}")
                print(f"Percentage of valid depth points: {(valid_depths.size / source_depth_o3d_np.size * 100):.1f}%")
            else:
                print("Warning: No valid depth points found in RGBD image!")
                print(f"Total points: {source_depth_o3d_np.size}")
                print(f"Depth value range: [{np.min(source_depth_o3d_np):.3f}, {np.max(source_depth_o3d_np):.3f}]")
                
            # Compute odometry
            success, trans, info = self.odometry.compute_odometry(
                source_rgbd, target_rgbd,
                self.intrinsic
            )
            
            frame_time = time.time() - start_time
            total_time += frame_time
            
            # Update visualizations
            if self.config['visualization']['show_correspondence']:
                self.frame_vis.update(
                    source_color, source_depth_m, 
                    target_color, target_depth_m,
                    trans if success else None,
                    self.intrinsic
                )
            
            if success:
                successful_frames += 1
                global_transform = np.dot(global_transform, trans)
                trajectory.append(global_transform.copy())
                self.trajectory_vis.add_point(global_transform)
                
                # Calculate frame-to-frame translation
                frame_translation = np.linalg.norm(trans[:3, 3])
                frame_translations.append(frame_translation)
                
                # Print odometry information
                translation = trans[:3, 3]
                rotation = trans[:3, :3]
                euler_angles = rotation_matrix_to_euler_angles(rotation)
                
                print(f"\nFrame {i:03d} -> {i+1:03d} transformation:")
                print(f"Translation (m): ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})")
                print(f"Translation magnitude: {frame_translation:.3f}m")
                print(f"Rotation (deg): ({euler_angles[0]:.2f}, {euler_angles[1]:.2f}, {euler_angles[2]:.2f})")
                print(f"Full transformation matrix:")
                print(trans)
                print(f"Processing time: {frame_time:.3f} seconds")
                
                if hasattr(info, 'fitness') and hasattr(info, 'inlier_rmse'):
                    print(f"Fitness: {info.fitness:.4f}, Inlier RMSE: {info.inlier_rmse:.4f}")
            else:
                print(f"\nOdometry {i:03d}-{i+1:03d} failed")
            
            # Small delay to allow visualization updates
            plt.pause(0.001)
        
        # Print performance statistics
        if len(rgb_files) > 0:
            avg_time = total_time / len(rgb_files)
            success_rate = successful_frames / len(rgb_files) * 100
            print(f"\nPerformance Statistics:")
            print(f"Average processing time per frame: {avg_time:.3f} seconds")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Total processing time: {total_time:.1f} seconds")
            
            if frame_translations:
                print("\nTrajectory Statistics:")
                print(f"Average frame-to-frame translation: {np.mean(frame_translations):.3f} meters")
                print(f"Min frame-to-frame translation: {np.min(frame_translations):.3f} meters")
                print(f"Max frame-to-frame translation: {np.max(frame_translations):.3f} meters")
                print(f"Total trajectory length: {sum(frame_translations):.3f} meters")
        
        plt.show(block=True)
        return trajectory

def main():
    rospy.init_node('rgbd_odometry_processor', anonymous=True)
    
    bag_path = "data/rosbags/test.bag"
    config_path = "config/odometry_config.yaml"
    output_dir = "./data"
    
    processor = RGBDOdometryProcessor(bag_path, config_path, output_dir)
    
    try:
        num_frames = processor.extract_images_from_bag()
        
        if num_frames > 0:
            trajectory = processor.compute_odometry()
            
            if running:  # Only save if not interrupted
                np.save(os.path.join(output_dir, "trajectory.npy"), trajectory)
                print(f"Saved trajectory with {len(trajectory)} poses")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import copy

class OdometryMethods:
    def __init__(self, config):
        self.config = config
        self.setup_parameters()
        
    def setup_parameters(self):
        """Setup parameters from config."""
        self.method = self.config['odometry']['method']
        # Create option object with default values
        self.rgbd_params = o3d.pipelines.odometry.OdometryOption()
        # Update with config values
        self.rgbd_params.depth_diff_max = float(self.config['odometry']['rgbd']['depth_diff_max'])
        self.rgbd_params.depth_min = float(self.config['odometry']['rgbd']['depth_min'])
        self.rgbd_params.depth_max = float(self.config['odometry']['rgbd']['depth_max'])
        
    def preprocess_point_cloud(self, pcd):
        """Preprocess point cloud for registration."""
        if len(pcd.points) < 10:  # Check if point cloud is too sparse
            return None
            
        # Remove invalid points (NaN or Inf)
        pcd = pcd.remove_non_finite_points()
        if len(pcd.points) < 10:
            return None
            
        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(pcd.points) < 10:
            return None
            
        # Downsample the point cloud
        voxel_size = float(self.config['odometry']['feature']['voxel_size'])
        pcd_down = pcd.voxel_down_sample(voxel_size)
        if len(pcd_down.points) < 10:
            return None
            
        # Estimate normals
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        return pcd_down
        
    def create_point_cloud(self, rgbd, intrinsic):
        """Create point cloud from RGBD image."""
        try:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                intrinsic,
                project_valid_depth_only=True
            )
            return pcd
        except Exception as e:
            print(f"Error creating point cloud: {e}")
            return None
        
    def point2point_odometry(self, source_rgbd, target_rgbd, intrinsic, init_transform=np.identity(4)):
        """Point-to-point RGBD odometry."""
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd, target_rgbd,
            intrinsic,
            init_transform,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
            self.rgbd_params
        )
        return success, trans, info
        
    def point2plane_odometry(self, source_rgbd, target_rgbd, intrinsic, init_transform=np.identity(4)):
        """Point-to-plane RGBD odometry."""
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd, target_rgbd,
            intrinsic,
            init_transform,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            self.rgbd_params
        )
        return success, trans, info
        
    def hybrid_odometry(self, source_rgbd, target_rgbd, intrinsic, init_transform=np.identity(4)):
        """Hybrid RGBD odometry (color + geometric)."""
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd, target_rgbd,
            intrinsic,
            init_transform,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            self.rgbd_params
        )
        return success, trans, info
        
    def hybrid_icp_odometry(self, source_rgbd, target_rgbd, intrinsic, init_transform=np.identity(4)):
        """Hybrid odometry with ICP refinement."""
        # First compute using hybrid method
        success, trans, info = self.hybrid_odometry(source_rgbd, target_rgbd, intrinsic, init_transform)
        
        if not success:
            print("Initial hybrid odometry failed")
            return success, trans, info
            
        # Create point clouds
        source_pcd = self.create_point_cloud(source_rgbd, intrinsic)
        target_pcd = self.create_point_cloud(target_rgbd, intrinsic)
        
        if source_pcd is None or target_pcd is None:
            print("Failed to create point clouds")
            return False, np.identity(4), None
            
        # Preprocess point clouds
        source_pcd_down = self.preprocess_point_cloud(source_pcd)
        target_pcd_down = self.preprocess_point_cloud(target_pcd)
        
        if source_pcd_down is None or target_pcd_down is None:
            print("Failed to preprocess point clouds")
            return False, np.identity(4), None
            
        # Apply initial transformation from hybrid method
        source_pcd_down.transform(trans)
        
        # ICP refinement
        icp_params = self.config['odometry']['icp']
        
        # Create ICP convergence criteria with proper type conversion
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria()
        criteria.max_iteration = int(icp_params['max_iteration'])
        criteria.relative_fitness = float(icp_params['relative_fitness'])
        criteria.relative_rmse = float(icp_params['relative_rmse'])
        
        try:
            # Run ICP
            result_icp = o3d.pipelines.registration.registration_icp(
                source_pcd_down, target_pcd_down,
                float(icp_params['max_correspondence_distance']),
                init_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria
            )
            
            if result_icp.fitness < 0.3:  # If ICP result is poor
                print(f"Poor ICP fitness: {result_icp.fitness}")
                return False, trans, info  # Return original hybrid result
                
            # Combine transformations
            final_trans = np.dot(result_icp.transformation, trans)
            return True, final_trans, result_icp
            
        except Exception as e:
            print(f"ICP registration failed: {e}")
            return False, trans, info  # Return original hybrid result
        
    def compute_odometry(self, source_rgbd, target_rgbd, intrinsic, init_transform=np.identity(4)):
        """Compute odometry using the configured method."""
        method_map = {
            'point2point': self.point2point_odometry,
            'point2plane': self.point2plane_odometry,
            'hybrid': self.hybrid_odometry,
            'hybrid_icp': self.hybrid_icp_odometry
        }
        
        if self.method not in method_map:
            raise ValueError(f"Unknown odometry method: {self.method}")
            
        return method_map[self.method](source_rgbd, target_rgbd, intrinsic, init_transform) 
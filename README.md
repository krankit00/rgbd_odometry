# RGBD Odometry with RealSense D435 and Open3D

This repository contains code to perform RGBD odometry using RealSense D435 camera data from a ROS bag file. The code extracts RGB and depth images from the bag file and uses Open3D to compute the camera trajectory.

## Prerequisites

- Python 3.6+
- ROS (tested with Noetic)
- Intel RealSense SDK 2.0

## Installation

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Modify the `bag_path` variable in `rgbd_odometry.py` to point to your ROS bag file:
```python
bag_path = "path_to_your_rosbag.bag"
```

2. Run the script:
```bash
python rgbd_odometry.py
```

The script will:
1. Extract RGB and depth images from the bag file into `./data/rgb` and `./data/depth` directories
2. Try to get camera intrinsics from the bag file (falls back to default D435 intrinsics if not available)
3. Compute RGBD odometry using Open3D
4. Save the trajectory as a NumPy array in `./data/trajectory.npy`

## ROS Topics

The script expects the following ROS topics in the bag file:
- RGB images: `/camera/color/image_raw`
- Depth images: `/camera/depth/image_rect_raw`

If your topics are different, modify the `rgb_topic` and `depth_topic` variables in the `extract_images_from_bag()` method.

## Output

- RGB images are saved as PNG files in `./data/rgb`
- Depth images are saved as 16-bit PNG files in `./data/depth`
- The camera trajectory is saved as a NumPy array in `./data/trajectory.npy`
  - Each entry in the trajectory is a 4x4 transformation matrix
  - The trajectory starts at the identity matrix
  - Each subsequent transformation is relative to the first frame

## Parameters

The odometry parameters can be adjusted in the `compute_odometry()` method:
- `depth_diff_max`: Maximum depth difference for correspondence (default: 0.07 meters)
- `depth_min`: Minimum depth threshold (default: 0.3 meters)
- `depth_max`: Maximum depth threshold (default: 3.0 meters)

## Notes

- The code uses the hybrid RGB-D odometry method from Park et al. (2017) which combines both photometric and geometric constraints
- If camera intrinsics cannot be read from the bag file, it falls back to default D435 intrinsics for 640x480 resolution
- Images are synchronized by finding the closest depth frame for each RGB frame 
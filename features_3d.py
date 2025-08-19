import numpy as np
import cv2
import open3d as o3d

def extract_3d_features(segmented_depth, mask):

    # Sanity check
    if segmented_depth.size == 0:
        raise ValueError("No pixels found in mask.")

    # Basic statistics
    max_depth = np.max(segmented_depth)
    min_depth = np.min(segmented_depth)
    avg_depth = np.mean(segmented_depth)
    std_depth = np.std(segmented_depth)
    depth_sum = np.sum(segmented_depth)
    depth_range = max_depth - min_depth
    distance_min_avg = np.abs(min_depth - avg_depth)
    distance_max_avg = np.abs(max_depth - avg_depth)

    # Projected area = number of foreground pixels (can be calibrated to real area if needed)
    projected_area = np.count_nonzero(mask)
    
        # Equation (4): approximate volume
    # Use numpy int64 to avoid overflow issues
    projected_area_int = np.int64(projected_area)
    max_depth_int = np.int64(max_depth)
    depth_sum_int = np.int64(depth_sum)
    
    # Calculate volume using int64 arithmetic
    volume_product = projected_area_int * max_depth_int
    approx_volume = volume_product - depth_sum_int
    


    return {
        "Feature17_ApproxVolume": approx_volume,
        "Feature18_MaxDepth": max_depth,
        "Feature19_MinDepth": min_depth,
        "Feature20_AvgDepth": avg_depth,
        "Feature21_DepthRange": depth_range,
        "Feature22_StdDepth": std_depth,
        "Feature23_SumDepth": depth_sum,
        "Feature24_MinMinusAvg": distance_min_avg,
        "Feature25_MaxMinusAvg": distance_max_avg,
    }


def compute_approx_volume_segmented_depth(segmented_depth):
    """
    Computes approximate volume from a segmented depth array where
    non-zero values represent the broiler's depth (e.g. in mm or meters).
    """
    # Convert to float for safety
    depth = segmented_depth.astype(np.float32)

    # Option 1: Background is 0
    depth_values = depth[depth > 0]

    # Option 2: Background is NaN (use if needed)
    # depth_values = depth[~np.isnan(depth)]

    if depth_values.size == 0:
        raise ValueError("No valid depth values found.")

    area = depth_values.size
    max_depth = np.max(depth_values)
    depth_sum = np.sum(depth_values)

    volume = area * max_depth - depth_sum
    return volume

def visualize_broiler_pcld(depth, output_path="broiler_pointcloud.ply"):
    """
    Generates and saves a 3D point cloud from a segmented depth .npy file
    using Intel RealSense D435i intrinsics (1280x720).
    
    Parameters:
        depth(npy): the .npy file containing segmented depth in mm.
        output_path (str): Path to save the resulting .ply point cloud.
    """

    # Load depth and convert to meters
    depth = depth.astype(np.float32) / 1000.0  # mm → m
    height, width = depth.shape

    # RealSense D435i intrinsics for 1280x720
    FX = 631.7247314453125
    FY = 631.7247314453125
    CX = 638.6820678710938
    CY = 352.1949157714844

    # Setup Open3D camera intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=width, height=height, fx=FX, fy=FY, cx=CX, cy=CY)
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (i - CX) * z / FX
    y = (j - CY) * z / FY
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    valid_mask = z.flatten() > 0
    points = points[valid_mask]/1000.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd],
        zoom=0.7,
        front=[0.0, 0.0, -1.0],
        lookat=[0.0, 0.0, 0.0],
        up=[0.0, -1.0, 0.0]
    )



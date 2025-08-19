"""
2D Feature Extraction Functions
Contains all functions for geometric features, convexity defects, and contour analysis
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict


def point_to_line_distance(point: Tuple[float, float], 
                         line_start: Tuple[float, float], 
                         line_end: Tuple[float, float]) -> float:
    """Calculate perpendicular distance from point to line segment."""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    if denominator == 0:
        return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    
    return numerator / denominator


def douglas_peucker(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    """Simplify a curve using Douglas-Peucker algorithm."""
    if len(points) <= 2:
        return points
    
    # Find point with maximum distance
    max_distance = 0
    max_index = 0
    
    for i in range(1, len(points) - 1):
        distance = point_to_line_distance(points[i], points[0], points[-1])
        if distance > max_distance:
            max_distance = distance
            max_index = i
    
    if max_distance > epsilon:
        left = douglas_peucker(points[:max_index + 1], epsilon)
        right = douglas_peucker(points[max_index:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def calculate_convexity_defects(contour: np.ndarray) -> Tuple[float, float]:
    """
    Calculate maximum and sum of convexity defect depths.
    
    Returns:
        Tuple of (max_defect, sum_defects)
    """
    if contour is None or len(contour) <= 3:
        return (0.0, 0.0)
    
    try:
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) <= 3:
            return (0.0, 0.0)
        
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return (0.0, 0.0)
        
        depths = defects[:, 0, 3] / 256.0
        return float(np.max(depths)), float(np.sum(depths))
        
    except Exception as e:
        print(f"Warning: Error calculating convexity defects: {e}")
        return (0.0, 0.0)


def extract_geometric_features(contour: np.ndarray, pixel_to_mm: float = 1.0) -> Dict[str, float]:
    """
    Extract all geometric features from a contour.
    
    Args:
        contour: OpenCV contour array
        pixel_to_mm: Conversion factor from pixels to millimeters
    
    Returns:
        Dictionary of features with their values
    """
    features = {}
    
    # 1. Projected area
    area = cv2.contourArea(contour) * (pixel_to_mm ** 2)
    features['projected_area'] = float(area)
    
    # 2. Contour perimeter
    perimeter = cv2.arcLength(contour, True) * pixel_to_mm
    features['perimeter'] = float(perimeter)
    
    # 3 & 4. Width and height
    x, y, w, h = cv2.boundingRect(contour)
    features['width'] = float(w * pixel_to_mm)
    features['height'] = float(h * pixel_to_mm)
    
    # 5. Convex hull area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) * (pixel_to_mm ** 2)
    features['convex_hull_area'] = float(hull_area)
    
    # 6, 7, 8. Ellipse features
    if len(contour) >= 5:
        (_, (major, minor), angle) = cv2.fitEllipse(contour)
        features['minor_axis_length'] = float(minor * pixel_to_mm)
        features['major_axis_length'] = float(major * pixel_to_mm)
        features['eccentricity'] = np.sqrt(1 - (minor / major) ** 2) if major > 0 else 0.0
    else:
        features['minor_axis_length'] = 0.0
        features['major_axis_length'] = 0.0
        features['eccentricity'] = 0.0
    
    # 9. Convex hull perimeter
    hull_perimeter = cv2.arcLength(hull, True) * pixel_to_mm
    features['convex_hull_perimeter'] = float(hull_perimeter)
    
    # 10 & 11. Approximate contour using Douglas-Peucker
    epsilon = 0.01 * perimeter
    points = [(p[0][0], p[0][1]) for p in contour]
    approx_points = douglas_peucker(points, epsilon)
    approx_contour = np.array([[[int(x), int(y)]] for x, y in approx_points], dtype=np.int32)
    
    features['approx_area'] = float(cv2.contourArea(approx_contour) * (pixel_to_mm ** 2))
    features['approx_perimeter'] = float(cv2.arcLength(approx_contour, True) * pixel_to_mm)
    
    # 12. Area ratio (contour/rectangle)
    rect_area = w * h * (pixel_to_mm ** 2)
    features['area_ratio_rect'] = float(area / rect_area if rect_area > 0 else 0.0)
    
    # 13. Area ratio (contour/hull)
    features['area_ratio_hull'] = float(area / hull_area if hull_area > 0 else 0.0)
    
    # 14 & 15. Convexity defects
    max_defect, sum_defects = calculate_convexity_defects(contour)
    features['max_convexity_defect'] = float(max_defect * pixel_to_mm)
    features['sum_convexity_defects'] = float(sum_defects * pixel_to_mm)
    
    # 16. Equivalent circle diameter
    features['equiv_diameter'] = float(2 * np.sqrt(area / np.pi))
    
    # 17. Approximate volume (assuming circular cross-section)
    features['approx_volume'] = float(area * np.sqrt(area / np.pi) / 3)
    
    return features


def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess image and extract largest contour.
    
    Returns:
        Tuple of (original_image, binary_image, largest_contour)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # Clean up binary image
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    return image, binary, largest_contour
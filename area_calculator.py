import cv2 as cv
import numpy as np
import sys
from scipy.spatial import KDTree

def darken_color(color, factor=0.7):
    return tuple((np.array(color, dtype=np.float32) * factor).astype(np.uint8))

def calculate_areas_from_outline(input_path, output_path=None, pixel_area=0.00004, background_output_path=None):
    # Read input raster (PNG)
    image = cv.imread(input_path, cv.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not read image {input_path}")
        sys.exit(1)

    # Convert to grayscale if needed
    if image.ndim == 3:
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Threshold to get binary mask (assume shapes are now white)
    _, binary = cv.threshold(image_gray, 127, 255, cv.THRESH_BINARY)

    # Find connected components
    numLabels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=4)

    print(f"Found {numLabels-1} connected components (excluding background).")
    total_area = 0

    # Prepare output image with random colors for each component
    output_img = np.zeros((*binary.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    colors = rng.integers(50, 255, size=(numLabels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # background is white
    colors[1] = [255, 255, 255]
    colors[2] = [0, 0, 0]

    for i in range(0, numLabels):
        output_img[labels == i] = colors[i]      
    color_img = output_img.copy()
    
    for i in range(3, numLabels):
        area = stats[i, cv.CC_STAT_AREA]
        total_area += area
        cX, cY = centroids[i]
        print(f"Component {i}: Area = {area} px = {area * pixel_area:.4f} m^2, Center = ({cX:.1f}, {cY:.1f})")
        cv.putText(output_img, f"{area * pixel_area:.2f} m^2", (int(cX - 40), int(cY)),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if output_path:
        cv.imwrite(output_path, output_img)
        print(f"Colored and annotated image saved to {output_path}")
    print(f"Total area of components: {total_area * pixel_area:.4f} m^2")
    
    

    # Write background-darkened image to a separate path if requested
    if background_output_path:
        black_mask = np.all(color_img == [0, 0, 0], axis=-1)
        coords_black = np.column_stack(np.where(black_mask))
        coords_colored = np.column_stack(np.where(~black_mask))
        colored_pixels = color_img[~black_mask]
        tree = KDTree(coords_colored)
        _, idxs = tree.query(coords_black, p=2)  # p=1 for Manhattan distance
        for i, (y, x) in enumerate(coords_black):
            nearest_color = colored_pixels[idxs[i]]
            color_img[y, x] = darken_color(nearest_color, factor=0.7)
        
        for i in range(3, numLabels):
            area = stats[i, cv.CC_STAT_AREA]
            total_area += area
            cX, cY = centroids[i]
            print(f"Component {i}: Area = {area} px = {area * pixel_area:.4f} m^2, Center = ({cX:.1f}, {cY:.1f})")
            cv.putText(color_img, f"{area * pixel_area:.2f} m^2", (int(cX - 40), int(cY)),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imwrite(background_output_path, color_img)
        print(f"Background-darkened image saved to {background_output_path}")    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python area_calculator.py <input_png> [output_png] [background_output_png]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    background_output_path = sys.argv[3] if len(sys.argv) > 3 else None
    calculate_areas_from_outline(input_path, output_path, background_output_path=background_output_path)
import cv2
import numpy as np
from sklearn.cluster import KMeans
import json
import os
import colorsys
from typing import List, Tuple, Dict, Any, Optional

DEFAULT_CONFIG = {
    "kmeans_clusters": 5,
    "mood_map": {
        "Energetic": {"hue_range": [(0, 30), (330, 360)], "saturation_min": 0.5, "value_min": 0.5},
        "Happy": {"hue_range": [(31, 60)], "saturation_min": 0.5, "value_min": 0.5},
        "Calm": {"hue_range": [(180, 270)], "saturation_min": 0.3, "value_min": 0.3},
        "Neutral": {"hue_range": [(0, 360)], "saturation_max": 0.3, "value_max": 0.7},
        "Mysterious": {"hue_range": [(271, 329)], "saturation_min": 0.4, "value_min": 0.2},
        "Nature": {"hue_range": [(61, 179)], "saturation_min": 0.3, "value_min": 0.3},
    },
    "resize_width": 320,
    "palette_vis_height": 50,
    "abstract_vis_size": (200, 200),
    "default_mood": "Unknown"
}

CONFIG_FILE = 'chromamood_config.json'

# --- Configuration Management ---

def load_config(filepath: str = CONFIG_FILE) -> Dict[str, Any]:
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
                # Ensure all default keys exist
                for key, value in DEFAULT_CONFIG.items():
                    config.setdefault(key, value)
                return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config file {filepath}: {e}. Using default config.")
            return DEFAULT_CONFIG.copy()
    else:
        print(f"Config file {filepath} not found. Using default config.")
        # Optionally save the default config here if desired
        # save_config(DEFAULT_CONFIG, filepath)
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any], filepath: str = CONFIG_FILE) -> None:
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {filepath}")
    except IOError as e:
        print(f"Error saving config file {filepath}: {e}")

# --- Color Processing ---

def get_dominant_colors(image: np.ndarray, k: int = 5, resize_width: Optional[int] = 100) -> List[Tuple[int, int, int]]:
    if image is None or image.size == 0:
        return []

    if resize_width:
        height, width, _ = image.shape
        aspect_ratio = height / width
        new_height = int(resize_width * aspect_ratio)
        img_resized = cv2.resize(image, (resize_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        img_resized = image

    # Reshape to a list of pixels (BGR)
    pixels = img_resized.reshape((-1, 3))
    pixels = np.float32(pixels) # Kmeans requires float32

    # Perform K-Means clustering
    if len(pixels) < k: # Handle cases with very few pixels
        k = max(1, len(pixels))

    if k == 0:
        return []

    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
    except ValueError: # Can happen if pixels array is empty after filtering etc.
         return []

    # Get the cluster centers (dominant colors in BGR)
    dominant_bgr_colors = kmeans.cluster_centers_.astype(int)

    # Convert BGR to RGB
    dominant_rgb_colors = [tuple(c[::-1]) for c in dominant_bgr_colors]

    return dominant_rgb_colors

def rgb_to_hsv(rgb_color: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = [x / 255.0 for x in rgb_color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s, v # Hue in degrees (0-360)

def get_average_hsv(colors_rgb: List[Tuple[int, int, int]]) -> Optional[Tuple[float, float, float]]:
    if not colors_rgb:
        return None

    hsv_colors = [rgb_to_hsv(c) for c in colors_rgb]

    # Average HSV needs careful handling for Hue (circular mean)
    # For simplicity here, we'll average components directly, which is less accurate for hue
    # A better approach involves converting H to vectors, averaging, then converting back.
    # Or, prioritize the most dominant color's hue. Let's use the first color's hue.
    
    avg_h = hsv_colors[0][0] # Use hue of the most dominant color
    avg_s = np.mean([c[1] for c in hsv_colors])
    avg_v = np.mean([c[2] for c in hsv_colors])

    return avg_h, avg_s, avg_v


# --- Mood Mapping ---

def get_mood_from_palette(
    colors_rgb: List[Tuple[int, int, int]],
    mood_map: Dict[str, Dict[str, Any]],
    default_mood: str = "Unknown"
) -> str:
    if not colors_rgb:
        return default_mood

    avg_hsv = get_average_hsv(colors_rgb)
    if avg_hsv is None:
        return default_mood

    avg_h, avg_s, avg_v = avg_hsv

    scores = {mood: 0 for mood in mood_map}

    # Score based on average HSV matching mood criteria
    for mood, criteria in mood_map.items():
        hue_match = False
        if "hue_range" in criteria:
            for h_min, h_max in criteria["hue_range"]:
                if h_min <= avg_h <= h_max:
                    hue_match = True
                    break
        else: # If no hue range specified, consider it a match
             hue_match = True

        sat_match = True
        if "saturation_min" in criteria and avg_s < criteria["saturation_min"]:
            sat_match = False
        if "saturation_max" in criteria and avg_s > criteria["saturation_max"]:
            sat_match = False

        val_match = True
        if "value_min" in criteria and avg_v < criteria["value_min"]:
            val_match = False
        if "value_max" in criteria and avg_v > criteria["value_max"]:
            val_match = False

        if hue_match and sat_match and val_match:
            scores[mood] += 1 # Simple scoring, could be weighted

    # Also consider individual dominant colors
    for color_rgb in colors_rgb:
        h, s, v = rgb_to_hsv(color_rgb)
        for mood, criteria in mood_map.items():
            hue_match = False
            if "hue_range" in criteria:
                for h_min, h_max in criteria["hue_range"]:
                    if h_min <= h <= h_max:
                        hue_match = True
                        break
            else:
                 hue_match = True

            sat_match = True
            if "saturation_min" in criteria and s < criteria["saturation_min"]:
                sat_match = False
            if "saturation_max" in criteria and s > criteria["saturation_max"]:
                sat_match = False

            val_match = True
            if "value_min" in criteria and v < criteria["value_min"]:
                val_match = False
            if "value_max" in criteria and v > criteria["value_max"]:
                val_match = False

            if hue_match and sat_match and val_match:
                 scores[mood] += 0.5 # Lower weight for individual colors


    # Find the mood with the highest score
    best_mood = default_mood
    max_score = 0
    # Sort moods alphabetically to break ties consistently
    sorted_moods = sorted(scores.keys())
    for mood in sorted_moods:
        if scores[mood] > max_score:
            max_score = scores[mood]
            best_mood = mood

    # If no mood scored above 0, return default
    if max_score <= 0:
        # Fallback: check average color against simplified warm/cool
        if avg_hsv:
            avg_h, avg_s, avg_v = avg_hsv
            if (0 <= avg_h <= 60 or 330 <= avg_h <= 360) and avg_s > 0.3 and avg_v > 0.4:
                return "Energetic" # Warm guess
            elif (180 <= avg_h <= 270) and avg_s > 0.3 and avg_v > 0.3:
                return "Calm" # Cool guess

        return default_mood


    return best_mood


# --- Data Visualization ---

def create_palette_image(
    colors_rgb: List[Tuple[int, int, int]],
    height: int = 50,
    width_per_color: int = 50
) -> Optional[np.ndarray]:
    if not colors_rgb:
        return None

    num_colors = len(colors_rgb)
    total_width = num_colors * width_per_color
    palette_img = np.zeros((height, total_width, 3), dtype=np.uint8)

    for i, color_rgb in enumerate(colors_rgb):
        # Convert RGB to BGR for OpenCV
        color_bgr = tuple(reversed(color_rgb))
        start_x = i * width_per_color
        end_x = (i + 1) * width_per_color
        cv2.rectangle(palette_img, (start_x, 0), (end_x, height), color_bgr, -1)

    return palette_img

def create_abstract_visual(
    colors_rgb: List[Tuple[int, int, int]],
    width: int = 200,
    height: int = 200,
    style: str = 'stripes' # 'stripes', 'blocks'
) -> Optional[np.ndarray]:
    if not colors_rgb:
        return None

    visual_img = np.zeros((height, width, 3), dtype=np.uint8)
    num_colors = len(colors_rgb)

    if style == 'stripes':
        stripe_height = height // num_colors if num_colors > 0 else height
        for i, color_rgb in enumerate(colors_rgb):
            color_bgr = tuple(reversed(color_rgb))
            start_y = i * stripe_height
            end_y = (i + 1) * stripe_height if i < num_colors - 1 else height
            cv2.rectangle(visual_img, (0, start_y), (width, end_y), color_bgr, -1)
    elif style == 'blocks':
        # Simple grid layout - adjust as needed
        cols = int(np.ceil(np.sqrt(num_colors)))
        rows = int(np.ceil(num_colors / cols))
        block_w = width // cols
        block_h = height // rows
        for i, color_rgb in enumerate(colors_rgb):
            color_bgr = tuple(reversed(color_rgb))
            row_idx = i // cols
            col_idx = i % cols
            start_x = col_idx * block_w
            start_y = row_idx * block_h
            end_x = start_x + block_w
            end_y = start_y + block_h
            cv2.rectangle(visual_img, (start_x, start_y), (end_x, end_y), color_bgr, -1)
    else: # Default to stripes
         stripe_height = height // num_colors if num_colors > 0 else height
         for i, color_rgb in enumerate(colors_rgb):
             color_bgr = tuple(reversed(color_rgb))
             start_y = i * stripe_height
             end_y = (i + 1) * stripe_height if i < num_colors - 1 else height
             cv2.rectangle(visual_img, (0, start_y), (width, end_y), color_bgr, -1)


    return visual_img

def draw_ui_elements(
    frame: np.ndarray,
    mood: str,
    palette_img: Optional[np.ndarray],
    abstract_img: Optional[np.ndarray],
    config: Dict[str, Any]
) -> None:
    frame_h, frame_w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255) # White
    bg_color = (0, 0, 0) # Black background for text
    thickness = 2
    line_type = cv2.LINE_AA

    # Display Mood
    mood_text = f"Mood: {mood}"
    (text_w, text_h), baseline = cv2.getTextSize(mood_text, font, font_scale, thickness)
    text_x = 10
    text_y = 30
    cv2.rectangle(frame, (text_x - 5, text_y - text_h - 5), (text_x + text_w + 5, text_y + baseline + 5), bg_color, -1)
    cv2.putText(frame, mood_text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)

    # Display Palette
    if palette_img is not None:
        pal_h, pal_w, _ = palette_img.shape
        pal_y_start = text_y + baseline + 15
        if pal_y_start + pal_h < frame_h and 10 + pal_w < frame_w:
             frame[pal_y_start:pal_y_start + pal_h, 10:10 + pal_w] = palette_img

    # Display Abstract Visual (optional)
    if abstract_img is not None:
        abs_h, abs_w, _ = abstract_img.shape
        abs_x_start = frame_w - abs_w - 10
        abs_y_start = 10
        if abs_y_start + abs_h < frame_h and abs_x_start > 0:
            frame[abs_y_start:abs_y_start + abs_h, abs_x_start:abs_x_start + abs_w] = abstract_img


# --- Metrics Calculation ---

def calculate_color_metrics(colors_rgb: List[Tuple[int, int, int]]) -> Dict[str, float]:
    if not colors_rgb:
        return {
            "average_hue": 0.0,
            "average_saturation": 0.0,
            "average_value": 0.0,
            "color_variance": 0.0 # Example metric
        }

    hsv_colors = [rgb_to_hsv(c) for c in colors_rgb]
    hues = np.array([c[0] for c in hsv_colors])
    saturations = np.array([c[1] for c in hsv_colors])
    values = np.array([c[2] for c in hsv_colors])

    # Simple average - less accurate for hue
    avg_h = np.mean(hues)
    avg_s = np.mean(saturations)
    avg_v = np.mean(values)

    # Example variance metric (variance of RGB values)
    rgb_np = np.array(colors_rgb)
    variance = np.mean(np.var(rgb_np, axis=0)) if len(rgb_np) > 0 else 0.0


    return {
        "average_hue": avg_h,
        "average_saturation": avg_s,
        "average_value": avg_v,
        "color_variance": variance
    }

# --- Webcam Utilities ---

def open_webcam(index: int = 0) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {index}")
        return None
    # Optional: Set resolution if needed, though this can fail silently
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def release_webcam(cap: Optional[cv2.VideoCapture]) -> None:
    if cap and cap.isOpened():
        cap.release()

# --- Main Execution Guard (for testing utils directly) ---
if __name__ == '__main__':
    print("ChromaMood Utilities Module")

    # Example usage (demonstration purposes)
    config = load_config()
    print("\nLoaded Configuration:")
    print(json.dumps(config, indent=2))

    # Test color conversion
    rgb_test = (255, 0, 0) # Red
    hsv_test = rgb_to_hsv(rgb_test)
    print(f"\nRGB {rgb_test} -> HSV {hsv_test}")

    # Test mood mapping (using average HSV)
    test_palette_warm = [(255, 100, 0), (240, 150, 30), (200, 50, 10)] # Warm colors
    test_palette_cool = [(0, 100, 255), (30, 150, 240), (10, 50, 200)] # Cool colors
    test_palette_muted = [(150, 150, 150), (100, 100, 100)] # Muted colors

    mood_warm = get_mood_from_palette(test_palette_warm, config['mood_map'], config['default_mood'])
    mood_cool = get_mood_from_palette(test_palette_cool, config['mood_map'], config['default_mood'])
    mood_muted = get_mood_from_palette(test_palette_muted, config['mood_map'], config['default_mood'])

    print(f"\nMood for warm palette: {mood_warm}")
    print(f"Mood for cool palette: {mood_cool}")
    print(f"Mood for muted palette: {mood_muted}")

    # Test metrics
    metrics_warm = calculate_color_metrics(test_palette_warm)
    print(f"\nMetrics for warm palette: {metrics_warm}")

    # Test visualization creation (won't display here, just creates arrays)
    palette_img = create_palette_image(test_palette_warm, height=config['palette_vis_height'])
    abstract_img = create_abstract_visual(test_palette_cool, width=100, height=100, style='blocks')

    if palette_img is not None:
        print(f"\nCreated palette image with shape: {palette_img.shape}")
        # To display: cv2.imshow("Palette", palette_img); cv2.waitKey(0); cv2.destroyAllWindows()
    if abstract_img is not None:
        print(f"Created abstract visual image with shape: {abstract_img.shape}")
        # To display: cv2.imshow("Abstract", abstract_img); cv2.waitKey(0); cv2.destroyAllWindows()

    # Test dominant color extraction (requires a dummy image)
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Fill with some colors
    dummy_image[0:50, 0:50] = (255, 0, 0) # Blue in BGR
    dummy_image[0:50, 50:100] = (0, 255, 0) # Green in BGR
    dummy_image[50:100, 0:50] = (0, 0, 255) # Red in BGR
    dummy_image[50:100, 50:100] = (200, 200, 50) # Teal-ish

    dominant_colors = get_dominant_colors(dummy_image, k=3, resize_width=None)
    print(f"\nDominant colors from dummy image (RGB): {dominant_colors}")

    print("\nUtils module tests complete.")
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import colorsys
import time

# --- Constants ---
K_CLUSTERS = 5
MOOD_PALETTE_HEIGHT = 70
MOOD_PALETTE_WIDTH_PER_COLOR = 100
RESIZE_WIDTH = 320 # Resize frame for faster processing

# --- Helper Functions ---

def get_dominant_colors(image, k=K_CLUSTERS):
    """
    Extracts dominant colors from an image using K-Means clustering.
    Args:
        image (np.ndarray): Input image (BGR format).
        k (int): Number of dominant colors to find.
    Returns:
        list: A list of dominant colors (BGR tuples), sorted by frequency.
              Returns None if clustering fails.
    """
    try:
        # Resize for speed
        height, width, _ = image.shape
        aspect_ratio = width / height
        new_height = int(RESIZE_WIDTH / aspect_ratio)
        resized_image = cv2.resize(image, (RESIZE_WIDTH, new_height), interpolation=cv2.INTER_AREA)

        # Reshape the image to be a list of pixels
        pixels = resized_image.reshape((-1, 3))

        # Convert to float32 for KMeans
        pixels = np.float32(pixels)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=k, n_init=4, random_state=0) # n_init reduced for speed
        kmeans.fit(pixels)

        # Get the cluster centers (dominant colors) and labels
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Count labels to find the frequency of each cluster
        label_counts = Counter(labels)

        # Sort colors by frequency (most frequent first)
        # Convert centers back to uint8 BGR
        dominant_colors_bgr = [centers[i].astype(np.uint8) for i in label_counts.keys()]
        sorted_indices = sorted(label_counts.keys(), key=lambda i: label_counts[i], reverse=True)
        sorted_dominant_colors = [centers[i].astype(np.uint8) for i in sorted_indices]

        return sorted_dominant_colors
    except Exception as e:
        print(f"Error during K-Means clustering: {e}")
        return None

def bgr_to_hsv_avg(bgr_colors):
    """Calculates the average HSV values from a list of BGR colors."""
    if not bgr_colors:
        return 0, 0, 0

    hsv_colors = []
    for bgr in bgr_colors:
        # Create a 1x1 pixel image to use cvtColor
        pixel_bgr = np.uint8([[bgr]])
        pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
        hsv_colors.append(pixel_hsv)

    hsv_colors = np.array(hsv_colors, dtype=np.float32)

    # Calculate average H, S, V
    # Hue average needs careful handling (circular mean), but simple average is often sufficient here
    avg_h = np.mean(hsv_colors[:, 0])
    avg_s = np.mean(hsv_colors[:, 1])
    avg_v = np.mean(hsv_colors[:, 2])

    return avg_h, avg_s, avg_v # H: 0-179, S: 0-255, V: 0-255

def map_color_to_mood(colors_bgr):
    """
    Maps a list of dominant BGR colors to a predefined mood.
    Args:
        colors_bgr (list): List of dominant BGR color tuples.
    Returns:
        str: The suggested mood string.
    """
    if not colors_bgr:
        return "Neutral"

    avg_h, avg_s, avg_v = bgr_to_hsv_avg(colors_bgr)

    # Normalize S and V to 0-1 range for easier thresholding
    avg_s_norm = avg_s / 255.0
    avg_v_norm = avg_v / 255.0

    # --- Mood Logic ---
    # Prioritize low saturation/value cases
    if avg_s_norm < 0.20 or avg_v_norm < 0.18:
        return "Muted / Neutral"
    if avg_v_norm < 0.4:
         return "Dark / Mysterious"

    # Hue is in range 0-179 (OpenCV)
    # Warm hues (Reds, Oranges, Yellows): ~0-30 and ~160-179
    is_warm = (avg_h <= 30 or avg_h >= 160)
    # Cool hues (Greens, Cyans, Blues): ~40-130
    is_cool = (40 <= avg_h <= 130)
    # Other hues (Yellow-Green, Violets, Magentas)

    if is_warm and avg_s_norm > 0.6 and avg_v_norm > 0.6:
        return "Energetic / Passionate"
    elif is_warm and avg_s_norm > 0.5 and avg_v_norm > 0.7:
         return "Warm / Happy"
    elif is_cool and avg_s_norm > 0.4 and avg_v_norm > 0.4:
        return "Calm / Relaxed"
    elif is_cool and avg_s_norm > 0.3:
        return "Cool / Serene"
    elif 130 < avg_h < 160 and avg_s_norm > 0.5: # Violets/Magentas
        return "Creative / Mysterious"
    elif 30 < avg_h < 40 and avg_s_norm > 0.6 and avg_v_norm > 0.6: # Yellow/Greens
        return "Cheerful / Fresh"

    # Fallback based on simpler properties
    if avg_v_norm > 0.75 and avg_s_norm > 0.5:
        return "Bright / Vibrant"
    if avg_s_norm < 0.35:
        return "Subtle / Neutral"

    return "Neutral" # Default fallback

def create_palette_visual(colors_bgr, height=MOOD_PALETTE_HEIGHT, width_per_color=MOOD_PALETTE_WIDTH_PER_COLOR):
    """Creates a horizontal bar image displaying the color palette."""
    num_colors = len(colors_bgr)
    if num_colors == 0:
        return np.zeros((height, width_per_color, 3), dtype=np.uint8)

    total_width = width_per_color * num_colors
    palette = np.zeros((height, total_width, 3), dtype=np.uint8)

    for i, color in enumerate(colors_bgr):
        start_col = i * width_per_color
        end_col = (i + 1) * width_per_color
        # Color needs to be a tuple for rectangle function
        bgr_tuple = tuple(map(int, color))
        cv2.rectangle(palette, (start_col, 0), (end_col, height), bgr_tuple, -1)

    return palette

def add_mood_text_to_palette(palette_img, mood_text):
    """Adds the mood text centered below the palette."""
    h, w, _ = palette_img.shape
    # Create a slightly larger canvas to add text below
    new_h = h + 40
    output_img = np.zeros((new_h, w, 3), dtype=np.uint8)
    output_img[0:h, 0:w] = palette_img # Copy palette to the top

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Mood: {mood_text}"
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h + 25 # Position below the palette

    cv2.putText(output_img, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return output_img

# --- Main Execution ---

def run_chromamood():
    """Initializes webcam and runs the ChromaMood analysis loop."""
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting ChromaMood...")
    print("Press 'q' to quit.")

    last_analysis_time = time.time()
    analysis_interval = 0.2 # seconds - analyze N times per second

    dominant_colors = None
    mood = "Initializing..."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            time.sleep(0.5)
            continue

        current_time = time.time()
        # --- Perform analysis periodically ---
        if current_time - last_analysis_time >= analysis_interval:
            last_analysis_time = current_time

            # Flip frame horizontally for a more intuitive mirror view
            frame_flipped = cv2.flip(frame, 1)

            # Get dominant colors
            dominant_colors = get_dominant_colors(frame_flipped, k=K_CLUSTERS)

            # Map colors to mood
            if dominant_colors:
                mood = map_color_to_mood(dominant_colors)
            else:
                mood = "Analysis Failed"

        # --- Visualization ---
        display_frame = cv2.flip(frame, 1) # Show the flipped frame

        # Create palette visual only if colors are available
        if dominant_colors:
            palette_vis = create_palette_visual(dominant_colors)
            palette_with_mood = add_mood_text_to_palette(palette_vis, mood)
            cv2.imshow('ChromaMood - Palette & Mood', palette_with_mood)
        else:
            # Show a placeholder if analysis hasn't run or failed
            placeholder_palette = np.zeros((MOOD_PALETTE_HEIGHT + 40, MOOD_PALETTE_WIDTH_PER_COLOR * K_CLUSTERS, 3), dtype=np.uint8)
            cv2.putText(placeholder_palette, f"Mood: {mood}", (10, MOOD_PALETTE_HEIGHT + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
            cv2.imshow('ChromaMood - Palette & Mood', placeholder_palette)


        # Add mood text to the main frame as well (optional)
        # cv2.putText(display_frame, f"Mood: {mood}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('ChromaMood - Live Feed', display_frame)

        # --- Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting ChromaMood.")
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_chromamood()
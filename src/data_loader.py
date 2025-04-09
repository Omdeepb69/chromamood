import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import time

def get_dominant_colors(image, k=5, image_processing_size=None):
    """
    Extracts dominant colors from an image using K-Means clustering.
    """
    if image is None:
        return []

    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, interpolation=cv2.INTER_AREA)

    # Reshape the image to be a list of pixels (BGR)
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Perform K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    try:
        compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
    except cv2.error as e:
        print(f"Error during K-Means: {e}")
        # Handle cases with very few colors, maybe return the most frequent pixel?
        # For simplicity, returning empty list for now
        return []


    centers = np.uint8(centers)

    # Count labels to find frequency of each cluster
    label_counts = np.bincount(labels.flatten())

    # Sort centers by frequency
    sorted_indices = np.argsort(label_counts)[::-1]
    dominant_colors = centers[sorted_indices]

    return dominant_colors # BGR format

def bgr_to_hsv(bgr_color):
    """Converts a single BGR color to HSV."""
    b, g, r = bgr_color
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    return h * 360, s * 100, v * 100 # Hue (0-360), Sat (0-100), Val (0-100)

def get_mood(colors_bgr, threshold_saturation=20, threshold_value=20):
    """
    Determines a mood based on the dominant colors (BGR format).
    """
    if not colors_bgr.any():
        return "Neutral"

    hues = []
    saturations = []
    values = []

    for color in colors_bgr:
        h, s, v = bgr_to_hsv(color)
        # Ignore very dark/desaturated colors for mood calculation
        if s > threshold_saturation and v > threshold_value:
            hues.append(h)
            saturations.append(s)
            values.append(v)

    if not hues: # All colors were below thresholds
        avg_value = np.mean([bgr_to_hsv(c)[2] for c in colors_bgr])
        if avg_value < 30:
            return "Dark / Somber"
        elif avg_value > 70:
             return "Bright / Neutral"
        else:
            return "Neutral / Grayish"


    avg_hue = np.mean(hues)
    avg_saturation = np.mean(saturations)
    avg_value = np.mean(values)

    # Simple Mood Logic based on Hue ranges
    if 0 <= avg_hue < 30 or 330 <= avg_hue <= 360: # Reds
        if avg_saturation > 50 and avg_value > 50:
            return "Passionate / Energetic"
        else:
            return "Warm / Muted Red"
    elif 30 <= avg_hue < 60: # Oranges/Yellows
         if avg_saturation > 50 and avg_value > 60:
            return "Happy / Cheerful"
         else:
            return "Warm / Earthy"
    elif 60 <= avg_hue < 150: # Greens
        if avg_saturation > 40 and avg_value > 40:
            return "Natural / Calm"
        else:
            return "Muted Green / Earthy"
    elif 150 <= avg_hue < 250: # Cyans/Blues
        if avg_saturation > 40 and avg_value > 40:
            return "Calm / Serene"
        else:
            return "Cool / Muted Blue"
    elif 250 <= avg_hue < 330: # Purples/Magentas
        if avg_saturation > 40 and avg_value > 40:
            return "Mysterious / Creative"
        else:
            return "Cool / Muted Purple"
    else:
        return "Neutral" # Should not happen with valid hues

def create_palette_visual(colors, height=50, width_per_color=50):
    """
    Creates a simple visual representation of the color palette.
    """
    num_colors = len(colors)
    if num_colors == 0:
        return np.zeros((height, width_per_color, 3), dtype=np.uint8)

    palette = np.zeros((height, width_per_color * num_colors, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        # color is BGR here
        palette[:, i * width_per_color:(i + 1) * width_per_color] = color
    return palette

def main():
    num_clusters = 5
    processing_width = 160 # Resize frame for faster processing

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate processing height maintaining aspect ratio
    aspect_ratio = frame_height / frame_width
    processing_height = int(processing_width * aspect_ratio)
    processing_size = (processing_width, processing_height)

    print("Starting ChromaMood...")
    print("Press 'q' to quit.")

    last_analysis_time = time.time()
    analysis_interval = 0.5 # seconds, analyze colors twice per second

    dominant_colors = np.array([])
    mood = "Initializing..."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        current_time = time.time()
        if current_time - last_analysis_time >= analysis_interval:
            # Analyze colors
            dominant_colors = get_dominant_colors(frame, k=num_clusters, image_processing_size=processing_size)
            mood = get_mood(dominant_colors)
            last_analysis_time = current_time

        # Create visuals
        palette_vis = create_palette_visual(dominant_colors, height=50, width_per_color=50)

        # Prepare display frame
        display_frame = frame.copy()
        palette_height, palette_width, _ = palette_vis.shape

        # Position palette at the bottom
        if palette_width <= frame_width:
             start_x = (frame_width - palette_width) // 2
             display_frame[frame_height - palette_height:frame_height, start_x:start_x + palette_width] = palette_vis
        else:
             # If palette wider than frame (unlikely with defaults), resize palette
             scaled_palette = cv2.resize(palette_vis, (frame_width, palette_height))
             display_frame[frame_height - palette_height:frame_height, 0:frame_width] = scaled_palette


        # Add Mood text
        text_position = (10, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (255, 255, 255) # White
        line_type = 2
        # Add black background for text readability
        text_size, _ = cv2.getTextSize(f"Mood: {mood}", font, font_scale, line_type)
        text_w, text_h = text_size
        cv2.rectangle(display_frame, (text_position[0] - 5, text_position[1] - text_h - 5), (text_position[0] + text_w + 5, text_position[1] + 5), (0,0,0), -1)
        cv2.putText(display_frame, f"Mood: {mood}", text_position, font, font_scale, font_color, line_type)


        # Display the resulting frame
        cv2.imshow('ChromaMood - Webcam Feed', display_frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("ChromaMood stopped.")

if __name__ == "__main__":
    main()
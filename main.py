import cv2
import numpy as np
from sklearn.cluster import KMeans
import argparse
import sys
import os
import colorsys
import time

# --- Configuration ---

DEFAULT_K_CLUSTERS = 5
RESIZE_WIDTH = 320 # Resize frame for faster processing
PALETTE_HEIGHT = 50
ART_WIDTH = 200
ART_HEIGHT = 200
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_COLOR = (255, 255, 255)
LINE_TYPE = 2

# --- Helper Functions ---

def extract_dominant_colors(image, k=DEFAULT_K_CLUSTERS):
    """
    Extracts dominant colors from an image using K-Means clustering.
    """
    try:
        # OpenCV uses BGR, KMeans expects float
        img_data = image.reshape((-1, 3))
        img_data = np.float32(img_data)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(img_data)

        # Get the cluster centers (dominant colors) and labels
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        # Calculate percentage of each color
        counts = np.bincount(labels)
        percentages = counts / len(labels)

        # Sort colors by percentage (descending)
        sorted_indices = np.argsort(percentages)[::-1]
        dominant_colors = colors[sorted_indices]
        dominant_percentages = percentages[sorted_indices]

        return dominant_colors, dominant_percentages

    except Exception as e:
        print(f"Error during color extraction: {e}", file=sys.stderr)
        return None, None

def get_color_properties(bgr_color):
    """Converts BGR color to HSV and returns properties."""
    b, g, r = bgr_color / 255.0  # Normalize to 0-1
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue_deg = h * 360
    saturation = s
    value = v
    return hue_deg, saturation, value

def map_colors_to_mood(colors, percentages):
    """
    Suggests a mood based on the dominant color palette properties.
    """
    if colors is None or len(colors) == 0:
        return "Undefined"

    avg_hue = 0
    avg_sat = 0
    avg_val = 0
    total_weight = 0

    for color, weight in zip(colors, percentages):
        hue, sat, val = get_color_properties(color)
        # Weighted average, handling hue wraparound is complex, simplify for now
        # A simple weighted average might skew results near 0/360 degrees
        # For simplicity, we'll use a weighted average of saturation and value
        avg_sat += sat * weight
        avg_val += val * weight
        total_weight += weight

    # Normalize averages (though total_weight should be close to 1)
    if total_weight > 0:
        avg_sat /= total_weight
        avg_val /= total_weight
    else:
        return "Undefined"

    # Determine dominant hue category based on the most dominant color
    dominant_hue, _, _ = get_color_properties(colors[0])

    # Mood mapping logic (simplified)
    if avg_val < 0.3:
        mood = "Gloomy / Mysterious"
    elif avg_sat < 0.2:
        mood = "Subdued / Calm"
    else:
        if 0 <= dominant_hue < 60 or 330 <= dominant_hue <= 360: # Reds/Oranges/Yellows
            if avg_sat > 0.6 and avg_val > 0.6:
                mood = "Energetic / Passionate"
            elif avg_val > 0.7:
                 mood = "Happy / Warm"
            else:
                mood = "Warm / Earthy"
        elif 60 <= dominant_hue < 150: # Greens/Cyans
            if avg_sat > 0.5:
                mood = "Fresh / Natural"
            else:
                mood = "Peaceful / Serene"
        elif 150 <= dominant_hue < 270: # Blues/Purples
            if avg_val < 0.5:
                mood = "Mysterious / Deep"
            elif avg_sat > 0.5:
                mood = "Cool / Calm"
            else:
                mood = "Relaxed / Serene"
        else: # Pinks/Magentas
             if avg_sat > 0.6 and avg_val > 0.6:
                 mood = "Playful / Vibrant"
             else:
                 mood = "Gentle / Romantic"

    # Refine based on overall brightness/saturation
    if avg_val > 0.8 and avg_sat > 0.7 and "Calm" not in mood and "Peaceful" not in mood:
        mood += " & Vibrant"
    elif avg_val < 0.4 and avg_sat < 0.3:
        mood = "Muted / Somber"


    return mood


def generate_abstract_art(colors, percentages, width=ART_WIDTH, height=ART_HEIGHT):
    """
    Generates a simple abstract art image (colored stripes) based on the palette.
    """
    if colors is None or len(colors) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    art_image = np.zeros((height, width, 3), dtype=np.uint8)
    current_x = 0
    for color, percentage in zip(colors, percentages):
        stripe_width = int(percentage * width)
        if stripe_width <= 0 and current_x < width: # Ensure even tiny percentages get some space if possible
             stripe_width = 1
        end_x = current_x + stripe_width
        # Clamp end_x to width to avoid errors
        end_x = min(end_x, width)
        # Ensure we draw something if current_x is still within bounds
        if current_x < width:
            cv2.rectangle(art_image, (current_x, 0), (end_x, height), color.tolist(), -1)
        current_x = end_x
        if current_x >= width:
            break

    # Fill any remaining space with the last color if rounding caused gaps
    if current_x < width:
         cv2.rectangle(art_image, (current_x, 0), (width, height), colors[-1].tolist(), -1)


    return art_image

def display_results(frame, colors, percentages, mood, art=None):
    """
    Displays the original frame, color palette, mood, and optional abstract art.
    """
    if colors is None:
        cv2.putText(frame, "Error processing colors", (10, 30), FONT, FONT_SCALE, (0, 0, 255), LINE_TYPE)
        cv2.imshow("ChromaMood", frame)
        return

    # Create palette bar
    palette_bar = np.zeros((PALETTE_HEIGHT, frame.shape[1], 3), dtype=np.uint8)
    current_x = 0
    for color, percentage in zip(colors, percentages):
        bar_width = int(percentage * frame.shape[1])
        if bar_width <= 0 and current_x < frame.shape[1]:
            bar_width = 1 # Min width of 1 pixel if possible
        end_x = current_x + bar_width
        end_x = min(end_x, frame.shape[1])
        if current_x < frame.shape[1]:
            cv2.rectangle(palette_bar, (current_x, 0), (end_x, PALETTE_HEIGHT), color.tolist(), -1)
        current_x = end_x
        if current_x >= frame.shape[1]:
            break
     # Fill any remaining space with the last color
    if current_x < frame.shape[1]:
         cv2.rectangle(palette_bar, (current_x, 0), (frame.shape[1], PALETTE_HEIGHT), colors[-1].tolist(), -1)


    # Add mood text to the palette bar
    text = f"Mood: {mood}"
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, LINE_TYPE)
    text_x = (palette_bar.shape[1] - text_width) // 2
    text_y = (PALETTE_HEIGHT + text_height) // 2

    # Add a dark background for text readability
    cv2.rectangle(palette_bar, (text_x - 5, text_y - text_height - baseline),
                  (text_x + text_width + 5, text_y + baseline), (0, 0, 0), -1)
    cv2.putText(palette_bar, text, (text_x, text_y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)

    # Combine frame and palette
    combined_display = cv2.vconcat([frame, palette_bar])

    # Display abstract art if available
    if art is not None:
        # Resize art to fit if needed, or create a separate window
        # Option 1: Separate Window
        cv2.imshow("Abstract Art", art)

        # Option 2: Combine (might require resizing frame or art)
        # This example uses a separate window for simplicity

    cv2.imshow("ChromaMood", combined_display)


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="ChromaMood: Analyze colors from webcam or image, suggest mood, and generate abstract art.")
    parser.add_argument("--source", default="0", help="Video source (webcam ID or path to image/video file). Default: 0 (default webcam).")
    parser.add_argument("-k", "--clusters", type=int, default=DEFAULT_K_CLUSTERS, help=f"Number of dominant colors to extract. Default: {DEFAULT_K_CLUSTERS}.")
    parser.add_argument("--art", action="store_true", help="Generate and display abstract art based on the color palette.")
    parser.add_argument("--width", type=int, default=RESIZE_WIDTH, help=f"Width to resize input for processing. Default: {RESIZE_WIDTH}.")

    args = parser.parse_args()

    source = args.source
    k = args.clusters
    generate_art = args.art
    resize_width = args.width

    if k <= 0:
        print("Error: Number of clusters (k) must be positive.", file=sys.stderr)
        sys.exit(1)

    # Determine if source is an image or video/webcam
    is_image = False
    if os.path.isfile(source):
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if any(source.lower().endswith(ext) for ext in img_extensions):
            is_image = True
        else:
            # Assume it's a video file if it exists but isn't an image
            print(f"Source '{source}' is not a recognized image file. Attempting to open as video.")

    if is_image:
        # --- Image Mode ---
        try:
            image = cv2.imread(source)
            if image is None:
                raise IOError(f"Could not read image file: {source}")

            # Resize image
            original_height, original_width = image.shape[:2]
            aspect_ratio = original_height / original_width
            new_height = int(resize_width * aspect_ratio)
            resized_image = cv2.resize(image, (resize_width, new_height), interpolation=cv2.INTER_AREA)

            print(f"Processing image: {source}")
            start_time = time.time()
            colors, percentages = extract_dominant_colors(resized_image, k)
            if colors is not None:
                mood = map_colors_to_mood(colors, percentages)
                art_image = None
                if generate_art:
                    art_image = generate_abstract_art(colors, percentages)

                end_time = time.time()
                print(f"Analysis complete in {end_time - start_time:.2f} seconds.")
                print(f"Dominant Colors (BGR): {colors.tolist()}")
                print(f"Suggested Mood: {mood}")

                display_results(resized_image, colors, percentages, mood, art_image)
                print("Press any key to exit.")
                cv2.waitKey(0)
            else:
                 print("Could not extract colors.", file=sys.stderr)


        except FileNotFoundError:
            print(f"Error: Image file not found at '{source}'", file=sys.stderr)
            sys.exit(1)
        except IOError as e:
            print(f"Error reading image file: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during image processing: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            cv2.destroyAllWindows()

    else:
        # --- Webcam/Video Mode ---
        try:
            # Try converting source to int for webcam ID
            video_source = int(source)
        except ValueError:
            # If not an int, assume it's a video file path
            video_source = source

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{source}'", file=sys.stderr)
            sys.exit(1)

        print(f"Starting real-time analysis from source: {source}")
        print("Press 'q' to quit.")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Info: End of video stream or cannot read frame.", file=sys.stderr)
                break

            # Resize frame
            original_height, original_width = frame.shape[:2]
            aspect_ratio = original_height / original_width
            new_height = int(resize_width * aspect_ratio)
            resized_frame = cv2.resize(frame, (resize_width, new_height), interpolation=cv2.INTER_AREA)

            # Process frame
            colors, percentages = extract_dominant_colors(resized_frame, k)
            mood = "Processing..."
            art_image = None
            if colors is not None:
                 mood = map_colors_to_mood(colors, percentages)
                 if generate_art:
                     art_image = generate_abstract_art(colors, percentages)

            display_results(resized_frame, colors, percentages, mood, art_image)

            frame_count += 1

            # Exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        end_time = time.time()
        total_time = end_time - start_time
        fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nAnalysis stopped. Processed {frame_count} frames in {total_time:.2f} seconds ({fps:.2f} FPS).")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
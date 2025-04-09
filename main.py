import cv2
import numpy as np
from sklearn.cluster import KMeans
import argparse
import sys
import os
import colorsys
import time
import random

# --- Configuration ---

DEFAULT_K_CLUSTERS = 5
RESIZE_WIDTH = 320  # Resize frame for faster processing
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
        if 0 <= dominant_hue < 60 or 330 <= dominant_hue <= 360:  # Reds/Oranges/Yellows
            if avg_sat > 0.6 and avg_val > 0.6:
                mood = "Energetic / Passionate"
            elif avg_val > 0.7:
                mood = "Happy / Warm"
            else:
                mood = "Warm / Earthy"
        elif 60 <= dominant_hue < 150:  # Greens/Cyans
            if avg_sat > 0.5:
                mood = "Fresh / Natural"
            else:
                mood = "Peaceful / Serene"
        elif 150 <= dominant_hue < 270:  # Blues/Purples
            if avg_val < 0.5:
                mood = "Mysterious / Deep"
            elif avg_sat > 0.5:
                mood = "Cool / Calm"
            else:
                mood = "Relaxed / Serene"
        else:  # Pinks/Magentas
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

def detect_facial_expression(face_img):
    """
    Detect facial expression (emotion) from a face image
    Using a simplified detection method based on geometric features
    """
    # We'll use a pre-trained Haar cascade for face detection
    # For emotion detection, we'll use a simplified approach based on facial landmarks
    
    # Load facial landmark detector from OpenCV's face module
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return "No face detected", None
    
    # For simplicity, process only the first face found
    (x, y, w, h) = faces[0]
    
    # Extract the face ROI (Region of Interest)
    face_roi = gray[y:y+h, x:x+w]
    
    # Use facial landmarks for a more accurate emotion detection
    # For this example, we'll use a simplified approach
    
    # Load facial landmark detector
    try:
        # Check if we can use DNN-based face detector and landmark predictor
        face_net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",  # You'll need these model files
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        # If files exist, use DNN-based detection
        use_dnn = True
    except:
        # Fall back to a simpler method if model files not available
        use_dnn = False
    
    # Since we may not have access to proper emotion detection models,
    # we'll use a simple heuristic based on face proportions
    
    # Calculate some basic facial measurements
    face_height = h
    face_width = w
    aspect_ratio = face_width / face_height if face_height > 0 else 0
    
    # Analyze brightness and contrast in different facial regions
    if face_roi.size > 0:
        # Upper face region (eyes, forehead)
        upper_face = face_roi[0:int(h*0.5), :]
        # Lower face region (mouth, chin)
        lower_face = face_roi[int(h*0.5):, :]
        
        # Calculate average intensity in each region
        if upper_face.size > 0 and lower_face.size > 0:
            upper_intensity = np.mean(upper_face)
            lower_intensity = np.mean(lower_face)
            intensity_ratio = upper_intensity / lower_intensity if lower_intensity > 0 else 1
            
            # Simple emotion classification based on region intensity ratios
            # This is a very simplified approach and won't be very accurate
            if intensity_ratio > 1.1:
                expression = "Surprised/Fearful"
            elif intensity_ratio < 0.9:
                expression = "Happy/Smiling"
            elif aspect_ratio > 0.85:
                expression = "Neutral/Calm"
            else:
                expression = "Serious/Focused"
        else:
            expression = "Unknown Expression"
    else:
        expression = "Unknown Expression"
    
    return expression, (x, y, w, h)

def generate_abstract_face_art(face_img, colors, percentages, width=ART_WIDTH, height=ART_HEIGHT):
    """
    Generates an abstract artistic representation of a face using the color palette.
    """
    if colors is None or len(colors) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a blank canvas
    art_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background color from palette
    bg_color = colors[0]
    cv2.rectangle(art_image, (0, 0), (width, height), bg_color.tolist(), -1)
    
    # Extract facial expression and face position
    expression, face_rect = detect_facial_expression(face_img)
    
    # If no face detected, create a general abstract art
    if face_rect is None:
        return generate_abstract_art(colors, percentages, width, height)
    
    # Use colors from the palette to create abstract face elements
    if len(colors) >= 2:
        # Extract common color elements
        primary_color = colors[0].tolist()
        secondary_color = colors[1].tolist()
        accent_colors = [c.tolist() for c in colors[2:]] if len(colors) > 2 else [primary_color]
        
        # Create abstract face shape based on detected emotion
        face_center_x = width // 2
        face_center_y = height // 2
        face_radius = min(width, height) // 3
        
        # Draw abstract face background/shape
        cv2.circle(art_image, (face_center_x, face_center_y), face_radius, secondary_color, -1)
        
        # Add abstract elements based on expression
        if "Happy" in expression or "Smiling" in expression:
            # Happy face has upward curving elements
            # Eyes
            eye_y = face_center_y - face_radius // 3
            left_eye_x = face_center_x - face_radius // 2
            right_eye_x = face_center_x + face_radius // 2
            eye_size = face_radius // 6
            
            cv2.circle(art_image, (left_eye_x, eye_y), eye_size, accent_colors[0], -1)
            cv2.circle(art_image, (right_eye_x, eye_y), eye_size, accent_colors[0], -1)
            
            # Smiling mouth
            mouth_y = face_center_y + face_radius // 3
            cv2.ellipse(art_image, 
                        (face_center_x, mouth_y), 
                        (face_radius // 2, face_radius // 4), 
                        0, 0, 180, 
                        accent_colors[-1], 
                        thickness=max(3, face_radius // 10))
            
        elif "Surprised" in expression or "Fearful" in expression:
            # Surprised face has circular elements
            # Wide eyes
            eye_y = face_center_y - face_radius // 3
            left_eye_x = face_center_x - face_radius // 2
            right_eye_x = face_center_x + face_radius // 2
            eye_size = face_radius // 4
            
            cv2.circle(art_image, (left_eye_x, eye_y), eye_size, accent_colors[0], -1)
            cv2.circle(art_image, (right_eye_x, eye_y), eye_size, accent_colors[0], -1)
            
            # O-shaped mouth
            mouth_y = face_center_y + face_radius // 2
            cv2.circle(art_image, (face_center_x, mouth_y), face_radius // 4, accent_colors[-1], -1)
            
        else:  # Neutral or other expressions
            # Neutral face has straight elements
            # Eyes
            eye_y = face_center_y - face_radius // 3
            left_eye_x = face_center_x - face_radius // 2
            right_eye_x = face_center_x + face_radius // 2
            eye_size = face_radius // 6
            
            cv2.circle(art_image, (left_eye_x, eye_y), eye_size, accent_colors[0], -1)
            cv2.circle(art_image, (right_eye_x, eye_y), eye_size, accent_colors[0], -1)
            
            # Straight mouth
            mouth_y = face_center_y + face_radius // 3
            cv2.line(art_image, 
                    (face_center_x - face_radius // 2, mouth_y), 
                    (face_center_x + face_radius // 2, mouth_y), 
                    accent_colors[-1], 
                    thickness=max(3, face_radius // 10))
        
        # Add some random artistic elements using the color palette
        for i in range(min(5, len(colors))):
            # Random position based on face position
            rand_x = random.randint(0, width-1)
            rand_y = random.randint(0, height-1)
            rand_size = random.randint(5, face_radius // 2)
            rand_color = colors[i % len(colors)].tolist()
            
            # Random shape (circle or rectangle)
            if random.random() > 0.5:
                cv2.circle(art_image, (rand_x, rand_y), rand_size, rand_color, -1)
            else:
                cv2.rectangle(art_image, 
                             (rand_x - rand_size, rand_y - rand_size), 
                             (rand_x + rand_size, rand_y + rand_size), 
                             rand_color, -1)
    
    # Add expression text to the image
    cv2.putText(art_image, f"Expression: {expression}", 
                (10, height - 10), FONT, FONT_SCALE * 0.8, 
                (255, 255, 255), LINE_TYPE)
    
    return art_image

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
        if stripe_width <= 0 and current_x < width:  # Ensure even tiny percentages get some space if possible
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

def display_results(frame, colors, percentages, mood, expression="Unknown", art=None):
    """
    Displays the original frame, color palette, mood, expression, and optional abstract art.
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
            bar_width = 1  # Min width of 1 pixel if possible
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

    # Add mood and expression text to the palette bar
    mood_text = f"Mood: {mood}"
    expression_text = f"Expression: {expression}"
    
    # Calculate positions for both texts
    (mood_width, text_height), baseline = cv2.getTextSize(mood_text, FONT, FONT_SCALE, LINE_TYPE)
    (expr_width, _), _ = cv2.getTextSize(expression_text, FONT, FONT_SCALE, LINE_TYPE)
    
    mood_x = 10
    expr_x = frame.shape[1] - expr_width - 10
    text_y = (PALETTE_HEIGHT + text_height) // 2

    # Add dark backgrounds for text readability
    cv2.rectangle(palette_bar, (mood_x - 5, text_y - text_height - baseline),
                  (mood_x + mood_width + 5, text_y + baseline), (0, 0, 0), -1)
    cv2.putText(palette_bar, mood_text, (mood_x, text_y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)
    
    cv2.rectangle(palette_bar, (expr_x - 5, text_y - text_height - baseline),
                  (expr_x + expr_width + 5, text_y + baseline), (0, 0, 0), -1)
    cv2.putText(palette_bar, expression_text, (expr_x, text_y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)

    # Combine frame and palette
    combined_display = cv2.vconcat([frame, palette_bar])

    # Display abstract art if available
    if art is not None:
        cv2.imshow("Abstract Art", art)

    cv2.imshow("ChromaMood", combined_display)


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="ChromaMood: Analyze colors from webcam or image, detect facial expressions, suggest mood, and generate abstract art.")
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
            
            # Detect facial expression
            expression, face_rect = detect_facial_expression(resized_image)
            
            if colors is not None:
                mood = map_colors_to_mood(colors, percentages)
                art_image = None
                if generate_art:
                    if face_rect is not None:
                        art_image = generate_abstract_face_art(resized_image, colors, percentages)
                    else:
                        art_image = generate_abstract_art(colors, percentages)

                end_time = time.time()
                print(f"Analysis complete in {end_time - start_time:.2f} seconds.")
                print(f"Dominant Colors (BGR): {colors.tolist()}")
                print(f"Suggested Mood: {mood}")
                print(f"Detected Expression: {expression}")

                # Draw face rectangle if detected
                if face_rect is not None:
                    x, y, w, h = face_rect
                    cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                display_results(resized_image, colors, percentages, mood, expression, art_image)
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
        print("Press 'q' to quit, 'c' to capture and create abstract face art.")

        frame_count = 0
        start_time = time.time()
        
        # For storing captured frame when 'c' is pressed
        captured_frame = None
        captured_colors = None
        captured_percentages = None

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
            expression = "Detecting..."
            art_image = None
            
            # Detect facial expression
            if colors is not None:
                expression, face_rect = detect_facial_expression(resized_frame)
                mood = map_colors_to_mood(colors, percentages)
                
                # Draw face rectangle if detected
                if face_rect is not None:
                    x, y, w, h = face_rect
                    cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if generate_art:
                    art_image = generate_abstract_art(colors, percentages)

            display_results(resized_frame, colors, percentages, mood, expression, art_image)

            frame_count += 1

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture current frame and colors for abstract face art
                if colors is not None:
                    print("Capturing frame and creating abstract face art...")
                    captured_frame = resized_frame.copy()
                    captured_colors = colors.copy()
                    captured_percentages = percentages.copy()
                    
                    # Generate and display abstract face art
                    face_art = generate_abstract_face_art(captured_frame, captured_colors, captured_percentages)
                    
                    if face_art is not None:
                        cv2.imshow("Captured Abstract Face Art", face_art)
                        
                        # Optional: Save the art
                        cv2.imwrite(f"face_art_{int(time.time())}.png", face_art)
                        print("Abstract face art saved!")
                else:
                    print("Cannot capture: No colors extracted from current frame.")

        end_time = time.time()
        total_time = end_time - start_time
        fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nAnalysis stopped. Processed {frame_count} frames in {total_time:.2f} seconds ({fps:.2f} FPS).")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
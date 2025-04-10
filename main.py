import cv2
import numpy as np
from sklearn.cluster import KMeans
import argparse
import sys
import os
import colorsys
import time
import random
import dlib
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import bz2

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

# Emotion labels for the pretrained model
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# --- Helper Functions ---

def download_models():
    """
    Downloads required models if they don't exist or uses local files.
    Returns model paths and a boolean indicating if models are available.
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    emotion_model_path = 'models/emotion_model.h5'
    shape_predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
    
    # Check if models exist and return paths
    if os.path.exists(emotion_model_path) and os.path.exists(shape_predictor_path):
        print("Using existing models found in 'models' directory.")
        return emotion_model_path, shape_predictor_path, True
    
    # Try to download shape predictor if it doesn't exist
    if not os.path.exists(shape_predictor_path):
        print("Downloading facial landmark detector...")
        # This URL is generally reliable
        url = 'https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2'
        compressed_path = 'models/shape_predictor_68_face_landmarks.dat.bz2'
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(compressed_path, 'wb') as f:
                    f.write(response.content)
                
                # Extract the bz2 file
                with open(shape_predictor_path, 'wb') as new_file, bz2.BZ2File(compressed_path, 'rb') as file:
                    for data in iter(lambda: file.read(100 * 1024), b''):
                        new_file.write(data)
                
                # Remove the compressed file
                os.remove(compressed_path)
                print("Facial landmark detector downloaded successfully.")
            else:
                print(f"Failed to download with status code: {response.status_code}")
                return None, None, False
        except Exception as e:
            print(f"Error downloading facial landmark detector: {e}")
            return None, None, False
    
    # Check for emotion model - we'll notify the user to download it manually
    if not os.path.exists(emotion_model_path):
        print("\nWARNING: Emotion recognition model not found!")
        print("To enable facial expression analysis, please manually download or create an emotion recognition model.")
        print("1. The model should be in Keras H5 format")
        print("2. It should be trained to recognize facial emotions into 7 categories (angry, disgust, fear, happy, sad, surprise, neutral)")
        print("3. Save it as 'models/emotion_model.h5'")
        print("\nRunning with limited functionality (color analysis only).")
        return None, None, False
    
    return emotion_model_path, shape_predictor_path, True

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

def detect_facial_expression(face_img, detector, predictor, emotion_model):
    """
    Detect facial expression (emotion) from a face image using dlib for landmark detection
    and a pretrained emotion recognition model.
    """
    # Convert to grayscale for detection
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using dlib
    faces = detector(gray)
    
    if len(faces) == 0:
        return "No face detected", None
    
    # Get the first face
    face = faces[0]
    
    # Get facial landmarks
    shape = predictor(gray, face)
    
    # Convert dlib rectangle to OpenCV rectangle format
    x = face.left()
    y = face.top()
    w = face.width()
    h = face.height()
    
    # Extract the face ROI for emotion detection
    roi = gray[y:y+h, x:x+w]
    
    try:
        # Resize ROI to match the input size of the emotion model
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Make a prediction on the ROI
        preds = emotion_model.predict(roi)[0]
        emotion_idx = np.argmax(preds)
        emotion_label = EMOTIONS[emotion_idx]
        confidence = preds[emotion_idx]
        
        # Only return the emotion if confidence is above threshold
        if confidence > 0.5:
            expression = emotion_label.capitalize()
        else:
            expression = "Neutral"  # Default to neutral if uncertain
            
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        expression = "Unknown"
    
    return expression, (x, y, w, h)

def generate_abstract_face_art(face_img, colors, percentages, expression, width=ART_WIDTH, height=ART_HEIGHT):
    """
    Generates an abstract artistic representation of a face using the color palette
    and detected expression.
    """
    if colors is None or len(colors) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a blank canvas
    art_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background color from palette
    bg_color = colors[0]
    cv2.rectangle(art_image, (0, 0), (width, height), bg_color.tolist(), -1)
    
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
        
        # Draw abstract face background/shape based on emotion
        if expression.lower() in ["angry", "disgust"]:
            # Jagged/angular face shape for angry/disgust
            points = np.array([
                [face_center_x - face_radius, face_center_y - face_radius],
                [face_center_x + face_radius, face_center_y - face_radius * 0.8],
                [face_center_x + face_radius * 0.9, face_center_y + face_radius],
                [face_center_x - face_radius * 0.9, face_center_y + face_radius * 0.9],
            ], dtype=np.int32)
            cv2.fillPoly(art_image, [points], secondary_color)
        elif expression.lower() in ["fear", "sad"]:
            # Downward-leaning oval for fear/sad
            axes = (face_radius, int(face_radius * 1.2))
            angle = 15 if expression.lower() == "fear" else 0
            cv2.ellipse(art_image, (face_center_x, face_center_y), axes, angle, 0, 360, secondary_color, -1)
        elif expression.lower() in ["happy", "surprise"]:
            # Rounded face for happy/surprise
            cv2.circle(art_image, (face_center_x, face_center_y), face_radius, secondary_color, -1)
        else:  # neutral or undefined
            # Square-ish face for neutral
            cv2.rectangle(art_image, 
                         (face_center_x - face_radius, face_center_y - face_radius),
                         (face_center_x + face_radius, face_center_y + face_radius),
                         secondary_color, -1)
            
        # Add abstract elements based on expression
        if expression.lower() == "happy":
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
            
        elif expression.lower() == "surprise":
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
            
        elif expression.lower() == "angry":
            # Angry face has angled elements
            # Angled eyebrows
            eye_y = face_center_y - face_radius // 3
            left_eye_x = face_center_x - face_radius // 2
            right_eye_x = face_center_x + face_radius // 2
            eye_size = face_radius // 6
            
            # Angled eyes
            cv2.ellipse(art_image, (left_eye_x, eye_y), 
                        (eye_size, eye_size // 2), 45, 0, 360, 
                        accent_colors[0], -1)
            cv2.ellipse(art_image, (right_eye_x, eye_y), 
                        (eye_size, eye_size // 2), 135, 0, 360, 
                        accent_colors[0], -1)
            
            # Downward frown
            mouth_y = face_center_y + face_radius // 3
            cv2.ellipse(art_image, 
                        (face_center_x, mouth_y + face_radius // 4), 
                        (face_radius // 2, face_radius // 4), 
                        0, 180, 360, 
                        accent_colors[-1], 
                        thickness=max(3, face_radius // 10))
            
        elif expression.lower() == "sad":
            # Sad face has downward elements
            # Droopy eyes
            eye_y = face_center_y - face_radius // 3
            left_eye_x = face_center_x - face_radius // 2
            right_eye_x = face_center_x + face_radius // 2
            eye_size = face_radius // 6
            
            cv2.ellipse(art_image, (left_eye_x, eye_y), 
                        (eye_size, eye_size // 2), 0, 0, 180, 
                        accent_colors[0], -1)
            cv2.ellipse(art_image, (right_eye_x, eye_y), 
                        (eye_size, eye_size // 2), 0, 0, 180, 
                        accent_colors[0], -1)
            
            # Downward frown more pronounced
            mouth_y = face_center_y + face_radius // 2
            cv2.ellipse(art_image, 
                        (face_center_x, mouth_y + face_radius // 3), 
                        (face_radius // 2, face_radius // 3), 
                        0, 180, 360, 
                        accent_colors[-1], 
                        thickness=max(3, face_radius // 10))
            
        elif expression.lower() == "fear":
            # Fearful face
            # Wide oval eyes
            eye_y = face_center_y - face_radius // 3
            left_eye_x = face_center_x - face_radius // 2
            right_eye_x = face_center_x + face_radius // 2
            eye_width = face_radius // 5
            eye_height = face_radius // 3
            
            cv2.ellipse(art_image, (left_eye_x, eye_y), 
                        (eye_width, eye_height), 0, 0, 360, 
                        accent_colors[0], -1)
            cv2.ellipse(art_image, (right_eye_x, eye_y), 
                        (eye_width, eye_height), 0, 0, 360, 
                        accent_colors[0], -1)
            
            # Irregular mouth shape
            mouth_y = face_center_y + face_radius // 3
            points = np.array([
                [face_center_x - face_radius // 3, mouth_y],
                [face_center_x, mouth_y + face_radius // 5],
                [face_center_x + face_radius // 3, mouth_y],
            ], dtype=np.int32)
            cv2.fillPoly(art_image, [points], accent_colors[-1])
            
        elif expression.lower() == "disgust":
            # Disgust face with asymmetric elements
            # Narrowed eyes
            eye_y = face_center_y - face_radius // 3
            left_eye_x = face_center_x - face_radius // 2
            right_eye_x = face_center_x + face_radius // 2
            eye_width = face_radius // 6
            eye_height = face_radius // 8
            
            cv2.ellipse(art_image, (left_eye_x, eye_y), 
                        (eye_width, eye_height), 0, 0, 360, 
                        accent_colors[0], -1)
            cv2.ellipse(art_image, (right_eye_x, eye_y), 
                        (eye_width, eye_height), 0, 0, 360, 
                        accent_colors[0], -1)
            
            # Curled up mouth on one side
            mouth_y = face_center_y + face_radius // 3
            cv2.line(art_image, 
                     (face_center_x - face_radius // 2, mouth_y),
                     (face_center_x + face_radius // 6, mouth_y),
                     accent_colors[-1], thickness=max(3, face_radius // 10))
            cv2.ellipse(art_image, 
                        (face_center_x + face_radius // 3, mouth_y - face_radius // 8), 
                        (face_radius // 6, face_radius // 8), 
                        0, 270, 90, 
                        accent_colors[-1], 
                        thickness=max(3, face_radius // 10))
            
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

def generate_art_from_frame(frame, k=DEFAULT_K_CLUSTERS, face_detection_enabled=False, detector=None, predictor=None, emotion_model=None, art_size=(ART_WIDTH, ART_HEIGHT)):
    """
    Helper function to generate art from a given frame.
    This consolidates the art generation code to be used in multiple places.
    """
    # Extract dominant colors
    colors, percentages = extract_dominant_colors(frame, k)
    
    # Create art based on detected face or simple abstract
    art = None
    expression = "Not analyzed"
    
    if face_detection_enabled:
        expression, face_rect = detect_facial_expression(frame, detector, predictor, emotion_model)
        
        # If face detected, generate face-based art
        if face_rect is not None:
            x, y, w, h = face_rect
            try:
                face_img = frame[y:y+h, x:x+w]
                art = generate_abstract_face_art(face_img, colors, percentages, expression, 
                                                width=art_size[0], height=art_size[1])
            except Exception as e:
                print(f"Error generating face art: {e}")
                art = generate_abstract_art(colors, percentages, 
                                           width=art_size[0], height=art_size[1])
        else:
            art = generate_abstract_art(colors, percentages, 
                                       width=art_size[0], height=art_size[1])
    else:
        art = generate_abstract_art(colors, percentages, 
                                   width=art_size[0], height=art_size[1])
    
    return art, colors, percentages, expression


# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description="ChromaMood: Analyze colors from webcam or image, detect facial expressions, suggest mood, and generate abstract art.")
    parser.add_argument("--source", default="0", help="Video source (webcam ID or path to image/video file). Default: 0 (default webcam).")
    parser.add_argument("-k", "--clusters", type=int, default=DEFAULT_K_CLUSTERS, help=f"Number of dominant colors to extract. Default: {DEFAULT_K_CLUSTERS}.")
    parser.add_argument("--art", action="store_true", help="Generate and display abstract art based on the color palette.")
    parser.add_argument("--width", type=int, default=RESIZE_WIDTH, help=f"Width to resize input for processing. Default: {RESIZE_WIDTH}.")
    parser.add_argument("--art-size", type=str, default=f"{ART_WIDTH}x{ART_HEIGHT}", help=f"Size of generated art in WIDTHxHEIGHT format. Default: {ART_WIDTH}x{ART_HEIGHT}.")

    args = parser.parse_args()

    source = args.source
    k = args.clusters
    generate_art = args.art
    resize_width = args.width
    
    # Continue from where the code stopped, completing the main() function
    try:
        # Parse art size parameter
        art_width, art_height = map(int, args.art_size.split('x'))
    except ValueError:
        print(f"Invalid art size format. Using default {ART_WIDTH}x{ART_HEIGHT}.")
        art_width, art_height = ART_WIDTH, ART_HEIGHT

    # Download or check for required models
    emotion_model_path, shape_predictor_path, models_available = download_models()
    
    # Initialize face detection and emotion recognition if models are available
    face_detection_enabled = False
    detector = None
    predictor = None
    emotion_model = None
    
    if models_available:
        try:
            # Initialize dlib face detector and shape predictor
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(shape_predictor_path)
            
            # Load emotion recognition model if available
            if emotion_model_path and os.path.exists(emotion_model_path):
                emotion_model = load_model(emotion_model_path)
                face_detection_enabled = True
                print("Face detection and emotion recognition enabled.")
            else:
                print("Emotion model not available. Running with color analysis only.")
        except Exception as e:
            print(f"Error initializing face detection: {e}")
            print("Running with color analysis only.")
    
    # Check if source is a webcam index, image file, or video file
    try:
        # Try to convert source to integer (webcam index)
        source = int(source)
    except ValueError:
        # Not an integer, treat as file path
        pass
    
    # Open video capture (webcam or video file)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}.")
        return
    
    # Get first frame to check if source is an image file
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from source.")
        cap.release()
        return
    
    # Check if source is a static image (certain file extensions)
    is_image = False
    if isinstance(source, str) and os.path.isfile(source):
        _, ext = os.path.splitext(source.lower())
        is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Process image or video
    while True:
        if not is_image:
            # Read new frame if not a static image
            ret, frame = cap.read()
            if not ret:
                # End of video or error reading frame
                break
        
        # Resize frame for faster processing
        if frame.shape[1] > resize_width:
            scale = resize_width / frame.shape[1]
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (resize_width, new_height))
        
        # Generate art, extract colors, and get expression
        art, colors, percentages, expression = generate_art_from_frame(
            frame, k, face_detection_enabled, detector, predictor, emotion_model, 
            art_size=(art_width, art_height)
        )
        
        # Map colors to mood
        mood = map_colors_to_mood(colors, percentages)
        
        # Display results
        display_results(frame, colors, percentages, mood, expression, art if generate_art else None)
        
        # Exit on 'q' key press
        key = cv2.waitKey(1 if not is_image else 0) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC key
            break
        
        # For static images, we only need to process once
        if is_image:
            # Keep window open until user exits
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                elif key == ord('s'):  # 's' key to save
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    # Save original image with palette
                    cv2.imwrite(f"chromamood_{timestamp}.jpg", 
                               cv2.vconcat([frame, cv2.resize(art, (frame.shape[1], art_height))]))
                    print(f"Saved as chromamood_{timestamp}.jpg")
                    if generate_art:
                        # Save art separately
                        cv2.imwrite(f"chromamood_art_{timestamp}.jpg", art)
                        print(f"Art saved as chromamood_art_{timestamp}.jpg")
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("ChromaMood ended.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in ChromaMood: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
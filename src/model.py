import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import colorsys
import os
import joblib
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score # Example metric, though less relevant here

# --- Constants ---
DEFAULT_N_CLUSTERS = 5
MODEL_SAVE_DIR = "saved_models"
KMEANS_MODEL_FILENAME = "kmeans_model.pkl" # Although we refit usually

# --- Helper Functions ---

def preprocess_image_for_kmeans(image):
    """Reshapes and converts image for K-Means."""
    # Check if image is grayscale, convert to BGR if so
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Check if image has alpha channel, remove it if so
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Reshape to a list of pixels (N, 3)
    pixels = image.reshape(-1, 3)
    # Convert to float32 for KMeans
    pixels = np.float32(pixels)
    return pixels

def rgb_to_hsv(rgb_color):
    """Converts an RGB color (0-255) tuple to HSV (H:0-360, S:0-1, V:0-1)."""
    r, g, b = [x / 255.0 for x in rgb_color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360, s, v)

def get_color_properties(colors_rgb):
    """Calculates average HSV properties for a list of RGB colors."""
    if not colors_rgb:
        return {'avg_hue': 0, 'avg_sat': 0, 'avg_val': 0, 'predominance': 'neutral'}

    hues = []
    sats = []
    vals = []
    for color in colors_rgb:
        h, s, v = rgb_to_hsv(color)
        hues.append(h)
        sats.append(s)
        vals.append(v)

    avg_hue = np.mean(hues)
    avg_sat = np.mean(sats)
    avg_val = np.mean(vals)

    # Determine hue predominance (simplified)
    warm_count = sum(1 for h in hues if (0 <= h < 60) or (300 <= h <= 360))
    cool_count = sum(1 for h in hues if 100 <= h < 260) # Greens, Blues, Violets

    if warm_count > cool_count:
        predominance = 'warm'
    elif cool_count > warm_count:
        predominance = 'cool'
    else:
        predominance = 'neutral' # Or mixed

    return {
        'avg_hue': avg_hue,
        'avg_sat': avg_sat,
        'avg_val': avg_val,
        'predominance': predominance
    }

# --- Core Model Functions ---

def train_kmeans(pixels, n_clusters=DEFAULT_N_CLUSTERS):
    """
    Fits a KMeans model to the image pixels.
    This isn't 'training' in the traditional sense, but fitting the clustering model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    return kmeans

def extract_dominant_colors(image, k=DEFAULT_N_CLUSTERS):
    """
    Extracts the k dominant colors from an image using KMeans.
    Returns colors in RGB format.
    """
    if image is None:
        return [], []

    pixels = preprocess_image_for_kmeans(image)
    if pixels.shape[0] == 0: # Handle empty images
        return [], []

    # Handle cases with fewer unique colors than k
    unique_colors = np.unique(pixels, axis=0)
    actual_k = min(k, len(unique_colors))

    if actual_k < 1: # No colors to cluster
        return [], []
    if actual_k == 1: # Only one color
         # Convert BGR (from OpenCV) to RGB
        dominant_colors_rgb = [tuple(int(c) for c in unique_colors[0][::-1])]
        percentages = [100.0]
        return dominant_colors_rgb, percentages


    kmeans = train_kmeans(pixels, n_clusters=actual_k)

    # Get the cluster centers (dominant colors in BGR)
    dominant_colors_bgr = kmeans.cluster_centers_.astype(int)

    # Convert BGR to RGB
    dominant_colors_rgb = [tuple(color[::-1]) for color in dominant_colors_bgr]

    # Calculate percentage of each color
    counts = Counter(kmeans.labels_)
    total_pixels = len(kmeans.labels_)
    percentages = [(counts[i] / total_pixels) * 100 for i in range(actual_k)]

    # Sort colors by percentage (descending)
    sorted_indices = np.argsort(percentages)[::-1]
    dominant_colors_rgb = [dominant_colors_rgb[i] for i in sorted_indices]
    percentages = [percentages[i] for i in sorted_indices]


    return dominant_colors_rgb, percentages


def suggest_mood_from_colors(colors_rgb):
    """Suggests a mood based on the dominant color properties."""
    if not colors_rgb:
        return "Neutral"

    properties = get_color_properties(colors_rgb)
    predominance = properties['predominance']
    avg_sat = properties['avg_sat']
    avg_val = properties['avg_val']

    mood = "Neutral" # Default

    if predominance == 'warm':
        if avg_sat > 0.6 and avg_val > 0.5:
            mood = "Energetic / Passionate"
        elif avg_sat > 0.4:
            mood = "Warm / Cozy"
        else:
            mood = "Mellow / Earthy"
    elif predominance == 'cool':
        if avg_sat > 0.5 and avg_val > 0.5:
            mood = "Refreshing / Vibrant"
        elif avg_sat > 0.3:
            mood = "Calm / Serene"
        else:
            mood = "Mysterious / Somber"
    else: # Neutral or mixed
        if avg_sat < 0.3 and avg_val < 0.4:
            mood = "Gloomy / Muted"
        elif avg_sat < 0.4:
            mood = "Subtle / Peaceful"
        elif avg_val > 0.6:
            mood = "Bright / Optimistic"

    # Refine based on brightness/saturation
    if avg_val < 0.3:
        mood += " (Dark Tones)"
    elif avg_val > 0.7 and avg_sat > 0.6:
         mood += " (Very Bright & Saturated)"
    elif avg_sat < 0.2:
        mood += " (Desaturated)"


    # Simple override examples
    # Count specific color types if needed for more complex rules
    is_mostly_blue = sum(1 for c in colors_rgb if rgb_to_hsv(c)[0] > 180 and rgb_to_hsv(c)[0] < 260) / len(colors_rgb) > 0.5
    is_mostly_red_orange = sum(1 for c in colors_rgb if rgb_to_hsv(c)[0] < 50 or rgb_to_hsv(c)[0] > 330) / len(colors_rgb) > 0.5

    if is_mostly_blue and avg_sat < 0.4:
        mood = "Deep Calm / Melancholy"
    if is_mostly_red_orange and avg_sat > 0.7:
        mood = "Intense Passion / Energy"


    return mood


def generate_abstract_art(colors_rgb, percentages, width=400, height=100):
    """Generates a simple abstract art image (color blocks/stripes)."""
    if not colors_rgb:
        return np.zeros((height, width, 3), dtype=np.uint8)

    art_image = np.zeros((height, width, 3), dtype=np.uint8)
    start_x = 0
    total_percentage = sum(percentages) # Ensure percentages sum roughly to 100

    if total_percentage == 0: # Avoid division by zero if percentages are empty/zero
         # Fill with the first color if available, otherwise black
        fill_color_bgr = colors_rgb[0][::-1] if colors_rgb else (0, 0, 0)
        cv2.rectangle(art_image, (0, 0), (width, height), fill_color_bgr, -1)
        return art_image

    for i, color_rgb in enumerate(colors_rgb):
        # Convert RGB to BGR for OpenCV drawing
        color_bgr = color_rgb[::-1]
        # Calculate width for this color block based on percentage
        block_width = int((percentages[i] / total_percentage) * width)
        end_x = start_x + block_width

        # Ensure the last block fills the remaining space
        if i == len(colors_rgb) - 1:
            end_x = width

        cv2.rectangle(art_image, (start_x, 0), (end_x, height), color_bgr, -1)
        start_x = end_x

    return art_image


# --- Model Evaluation (Conceptual Example) ---
# Note: Evaluating K-Means for color quantization is often subjective/visual.
# Silhouette score can give a rough idea of cluster separation, but might not
# correlate perfectly with perceived color accuracy.

def evaluate_kmeans_clustering(pixels, max_k=10):
    """Evaluates K-Means performance using Silhouette Score for different k."""
    results = {}
    if pixels.shape[0] < 2: # Need at least 2 samples for silhouette
        print("Not enough data points to evaluate.")
        return results

    print(f"Evaluating K-Means for k=2 to {max_k}...")
    for k in range(2, max_k + 1):
         # Ensure k is not more than number of unique points - 1
        unique_colors = np.unique(pixels, axis=0)
        if k >= len(unique_colors):
            print(f"Skipping k={k}, not enough unique points ({len(unique_colors)})")
            continue

        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            # Silhouette score requires at least 2 labels and points > n_clusters
            if len(set(labels)) > 1 and len(pixels) > k :
                 score = silhouette_score(pixels, labels, metric='euclidean')
                 results[k] = score
                 print(f"  k={k}: Silhouette Score = {score:.4f}")
            else:
                 print(f"  k={k}: Cannot calculate Silhouette Score (num_labels={len(set(labels))})")
                 results[k] = None

        except Exception as e:
            print(f"Error evaluating k={k}: {e}")
            results[k] = None
    return results

# --- Hyperparameter Tuning (Conceptual Example) ---
# For K-Means, the main hyperparameter is n_clusters (k).
# We can use evaluation metrics like silhouette score or elbow method (inertia)
# to help choose k, although for color quantization, a fixed k is common.

def tune_kmeans_hyperparameters(pixels, k_values=[3, 5, 7, 10]):
    """Finds the best 'k' based on Silhouette Score."""
    best_k = DEFAULT_N_CLUSTERS
    best_score = -1 # Silhouette score ranges from -1 to 1

    print(f"Tuning K-Means 'n_clusters' using Silhouette Score for k={k_values}...")
    if pixels.shape[0] < 2:
        print("Not enough data points to tune.")
        return best_k

    for k in k_values:
        unique_colors = np.unique(pixels, axis=0)
        if k >= len(unique_colors):
            print(f"Skipping k={k}, not enough unique points ({len(unique_colors)})")
            continue

        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)

            if len(set(labels)) > 1 and len(pixels) > k:
                score = silhouette_score(pixels, labels, metric='euclidean')
                print(f"  k={k}: Silhouette Score = {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = k
            else:
                 print(f"  k={k}: Cannot calculate Silhouette Score (num_labels={len(set(labels))})")

        except Exception as e:
            print(f"Error evaluating k={k}: {e}")

    print(f"Best k found: {best_k} with score: {best_score:.4f}")
    return best_k


# --- Model Saving/Loading (Less common for refitting K-Means, but possible) ---

def save_kmeans_model(kmeans_model, filename=KMEANS_MODEL_FILENAME):
    """Saves a trained KMeans model."""
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    filepath = os.path.join(MODEL_SAVE_DIR, filename)
    try:
        joblib.dump(kmeans_model, filepath)
        print(f"KMeans model saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving KMeans model: {e}")
        return False

def load_kmeans_model(filename=KMEANS_MODEL_FILENAME):
    """Loads a saved KMeans model."""
    filepath = os.path.join(MODEL_SAVE_DIR, filename)
    if os.path.exists(filepath):
        try:
            kmeans_model = joblib.load(filepath)
            print(f"KMeans model loaded from {filepath}")
            return kmeans_model
        except Exception as e:
            print(f"Error loading KMeans model: {e}")
            return None
    else:
        print(f"Model file not found: {filepath}")
        return None

# --- Prediction/Inference Function ---

def predict_mood_from_image(image, k=DEFAULT_N_CLUSTERS):
    """
    Performs the full pipeline: dominant color extraction and mood suggestion.
    """
    if image is None:
        return [], "No Image", None

    dominant_colors_rgb, percentages = extract_dominant_colors(image, k=k)

    if not dominant_colors_rgb:
        return [], "Could not extract colors", None

    suggested_mood = suggest_mood_from_colors(dominant_colors_rgb)
    abstract_art = generate_abstract_art(dominant_colors_rgb, percentages)

    return dominant_colors_rgb, suggested_mood, abstract_art

# --- Main Execution Example (for testing this module) ---

if __name__ == "__main__":
    print("ChromaMood Model Module - Running Test")

    # 1. Create a dummy image for testing
    test_image = np.zeros((200, 300, 3), dtype=np.uint8)
    # Add some color blocks (BGR format for cv2)
    test_image[0:100, 0:150] = [255, 0, 0]  # Blue
    test_image[0:100, 150:300] = [0, 255, 0] # Green
    test_image[100:200, 0:100] = [0, 0, 255]   # Red
    test_image[100:200, 100:200] = [0, 255, 255] # Yellow
    test_image[100:200, 200:300] = [200, 200, 200] # Gray

    print("\nTesting with a sample image...")
    dominant_colors, mood, art = predict_mood_from_image(test_image, k=5)

    print(f"Dominant Colors (RGB): {dominant_colors}")
    print(f"Suggested Mood: {mood}")

    if art is not None:
        print("Generated Abstract Art (showing dimensions):", art.shape)
        # Display the art if possible (requires GUI environment)
        try:
            cv2.imshow("Test Image", test_image)
            cv2.imshow("Generated Art", art)
            print("Displaying test image and generated art. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
             print(f"Could not display images (likely no GUI available): {e}")
    else:
        print("Art generation failed.")

    # 2. Test Evaluation and Tuning (using the test image pixels)
    print("\nTesting Evaluation & Tuning...")
    test_pixels = preprocess_image_for_kmeans(test_image)
    if test_pixels.shape[0] > 1:
        # evaluation_results = evaluate_kmeans_clustering(test_pixels, max_k=6)
        # print("Evaluation Results (Silhouette Score):", evaluation_results)

        # best_k = tune_kmeans_hyperparameters(test_pixels, k_values=[2, 3, 4, 5])
        # print(f"Tuning suggests best k = {best_k}")

        # Re-run prediction with suggested k (if different)
        # if best_k != DEFAULT_N_CLUSTERS:
        #    print(f"\nRe-running prediction with k={best_k}...")
        #    dominant_colors, mood, art = predict_mood_from_image(test_image, k=best_k)
        #    print(f"Dominant Colors (RGB): {dominant_colors}")
        #    print(f"Suggested Mood: {mood}")
        # Note: Evaluation/Tuning might not be very meaningful for this simple test image.
        #       It's more useful on complex, real-world images.
        print("Skipping detailed evaluation/tuning display for simple test image.")

    else:
        print("Skipping evaluation/tuning due to insufficient data points.")


    # 3. Test Model Saving/Loading (conceptual)
    # print("\nTesting Model Save/Load...")
    # dummy_kmeans = KMeans(n_clusters=3).fit(test_pixels[:10]) # Fit a tiny model
    # save_success = save_kmeans_model(dummy_kmeans, "test_model.pkl")
    # if save_success:
    #     loaded_model = load_kmeans_model("test_model.pkl")
    #     if loaded_model:
    #         print("Model successfully saved and loaded.")
    #         # Clean up test file
    #         try:
    #             os.remove(os.path.join(MODEL_SAVE_DIR, "test_model.pkl"))
    #             # Try removing directory if empty, ignore error if not empty
    #             os.rmdir(MODEL_SAVE_DIR)
    #         except OSError:
    #             pass # Directory might not be empty if other models exist
    #     else:
    #         print("Model loading failed.")
    # else:
    #     print("Model saving failed.")
    print("Skipping save/load test in this example run.")


    print("\nChromaMood Model Module Test Complete.")
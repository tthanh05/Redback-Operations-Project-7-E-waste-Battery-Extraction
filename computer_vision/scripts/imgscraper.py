import os
import ffmpeg
import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pytubefix import YouTube
from pytubefix import Playlist
from pytubefix.cli import on_progress
import re
import unicodedata
# --- CONFIG ---
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLdDkfYYZfGNYC5-HmpokPXi3U1OYrf-U1"
#YOUTUBE_URL = "https://www.youtube.com/watch?v=P8OZYabd22w"
OUTPUT_DIR = "../dataset/scraped"
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")
MASK_DIR = os.path.join(OUTPUT_DIR, "masked")
SAM_CHECKPOINT = os.path.join(os.path.dirname(__file__), "sam_vit_b_01ec64.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
video_path = ""
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

def make_filename_safe(filename):
    """
    Convert a filename into a filesystem-friendly version.

    Examples:
        "my file!.txt"           -> "my_file.txt"
        "résumé 2025.pdf"        -> "resume_2025.pdf"
        "data@#$%^&*().csv"      -> "data.csv"
        "hello   world!!.mp4"    -> "hello_world.mp4"
    """

    # Separate filename from extension
    parts = filename.rsplit('.', 1)

    if len(parts) == 2:
        name, ext = parts
        ext = "." + ext
    else:
        name = parts[0]
        ext = ""

    # Normalize unicode characters (é -> e)
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')

    # Replace spaces and invalid chars with underscores
    name = re.sub(r'[^A-Za-z0-9]+', '_', name)

    # Remove leading/trailing underscores
    name = name.strip('_')

    # Prevent empty filenames
    if not name:
        name = "file"

    return name

# --- STEP 1: Download video ---
def download_video(yt):
    print("[INFO] Downloading video...")
    ys = yt.streams.filter(file_extension="mp4", only_video=True).order_by('resolution').desc().first()
    fname = make_filename_safe(yt.title) + ".mp4"
    print(f"\nHEIGHT {ys.height}\n")
    ys.download(output_path=OUTPUT_DIR, filename=fname)
    
    # Sanitize title for filesystem and return the actual filename
    
    video_path = os.path.join(OUTPUT_DIR, fname)
    
    print(f"[INFO] Downloaded: {video_path}")
    return (yt.title, video_path)  # Return the actual path

# --- STEP 2: Extract frames (scene detection) ---
def extract_frames(video_path, title):
    print("[INFO] Extracting frames...")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found at {video_path}")
        return
    
    try:
        output_pattern = os.path.join(FRAME_DIR, f"{title}_frame_%04d.png")
        
        # Build ffmpeg command using ffmpeg-python
        (
            ffmpeg
            .input(video_path)
            .filter('select', 'gt(scene,0.3)')
            .output(output_pattern, vsync='vfr')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Count extracted frames
        frame_count = len([f for f in os.listdir(FRAME_DIR) if f.endswith('.png')])
        print(f"[INFO] Extracted {frame_count} frames")
        
    except ffmpeg.Error as e:
        print(f"[ERROR] ffmpeg failed: {e.stderr.decode() if e.stderr else str(e)}")


# --- STEP 3: Blur filter ---
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

# --- STEP 4: Load SAM ---
def load_sam():
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    return SamPredictor(sam)

def segment_with_prompt(predictor, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    h, w = image.shape[:2]

    # 👇 center point
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])  # 1 = foreground

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    # pick best mask
    best_mask = masks[np.argmax(scores)]

    return best_mask

# --- STEP 5: Segment + save transparent PNG ---
def segment_and_save(predictor):
    print("[INFO] Running segmentation...")
    
    for fname in sorted(os.listdir(FRAME_DIR)):
        if not fname.endswith('.png'):
            continue
            
        path = os.path.join(FRAME_DIR, fname)
        image = cv2.imread(path)

        if image is None:
            continue

        if is_blurry(image):
            print(f"[SKIP] Blurry frame: {fname}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = segment_with_prompt(predictor, image)

        # Apply mask → RGBA
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = image
        rgba[..., 3] = mask.astype(np.uint8) * 255

        # Crop to bounding box
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            print(f"[SKIP] Empty mask: {fname}")
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        cropped = rgba[y_min:y_max, x_min:x_max]

        out_path = os.path.join(MASK_DIR, fname.replace(".png", "_masked.png"))
        Image.fromarray(cropped).save(out_path)
        print(f"[SAVE] {out_path}")


# --- RUN PIPELINE ---
if __name__ == "__main__":
    pl = Playlist(PLAYLIST_URL)
    for video in pl.videos:
        title, video_path = download_video(video)
        extract_frames(video_path, make_filename_safe(title))
        predictor = load_sam()
        segment_and_save(predictor)

    print("[DONE] Pipeline complete.")
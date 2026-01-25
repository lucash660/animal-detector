import rawpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from pathlib import Path
import json
import shutil
import os

import birder
from birder.inference.classification import infer_image

DRAW = False
BLUR = True
ANIMAL_DETECTION = True

MAPPING_FILE = Path("./inat21-mapping.json") # Path("/Users/hanshaag/Downloads/inat21-mapping.json")
INPUT_FOLDER = Path("/Users/hanshaag/Documents/100MSDCF")
OUTPUT_FOLDER = Path("/Users/hanshaag/Documents/100MSDCF")

# Function to compute Laplacian variance as focus score
def focus_score(gray_np):
    # approximate Laplacian using numpy gradients
    gy, gx = np.gradient(gray_np.astype(float))
    laplacian = gx ** 2 + gy ** 2
    return laplacian.var()

def check_blurryiness(mask):

    # --- focus score only inside animal shape ---
    masked_gray = gray.copy().astype(float)
    masked_gray[~mask] = 0.0

    score = focus_score(masked_gray)
    status = "In Focus" if score > 10000 else "Blurry"

    return score, status

def infer_species(crop):
    crop_pil = Image.fromarray(crop)
    (out, _) = infer_image(net, crop_pil, transform)
    # --- species predictions ---

    # Convert out to numpy if it's a torch tensor
    if not isinstance(out, np.ndarray):
        out = out.cpu().numpy()  # if torch tensor

    return out

def draw_polygon(poly_pts):
    color = (0, 255, 0) if status == "In Focus" else (255, 0, 0)
    draw.line(poly_pts + [poly_pts[0]], fill=color, width=2)

    # --- label ---
    label = f"{score:.1f}"
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    lx, ly = poly_pts[0]
    ly = max(0, ly - th)

    draw.rectangle([lx, ly, lx + tw, ly + th], fill=color)
    draw.text((lx, ly), label, fill=(255, 255, 255), font=font)

# -------- Load iNat model --------
(net, model_info) = birder.load_pretrained_model(
    "vit_reg4_m16_rms_avg_i-jepa-inat21",
    inference=True
)

size = birder.get_size_from_signature(model_info.signature)
transform = birder.classification_transform(size, model_info.rgb_stats)

# -------- Load RAW fotos --------
files = list(INPUT_FOLDER.glob("*.ARW"))

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# -------- Load iNat classes --------
num_classes = 10000
INAT_CLASSES = [
    data.get(str(i), f"UNKNOWN_CLASS_{i}")
    for i in range(num_classes)
]

for file in files:

    with rawpy.imread(str(file)) as raw:
        rgb = raw.postprocess()

    # -------- Load detection model --------
    model = YOLO('yolov8n-seg.pt')
    results = model(rgb)
    
    if not results[0].masks: 
        species_folder = OUTPUT_FOLDER / "Unidentified"
        species_folder.mkdir(parents=True, exist_ok=True)  # create folder if needed

        # Copy the file
        dest_file = species_folder / file.name  # keep same name
        shutil.move(file, dest_file)
        continue
    
    class_names = model.names  # YOLO class names


    # Create a PIL image for drawing
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    # Optional: choose a font
    try:
        font = ImageFont.truetype("arial.ttf", 80)
    except:
        font = ImageFont.load_default(size=80)

    H, W = rgb.shape[:2]
    gray = np.array(Image.fromarray(rgb).convert("L"))

    animal_scores = []
    detections = []
    # -------- Loop over detections --------
    for box, polygon in zip(results[0].boxes, results[0].masks.xy):

        # --- build FULL-SIZE mask from polygon ---
        mask_img = Image.new("1", (W, H), 0)  # full-resolution binary mask
        mask_draw = ImageDraw.Draw(mask_img)

        poly_pts = [(int(x), int(y)) for x, y in polygon]
        mask_draw.polygon(poly_pts, outline=1, fill=1)

        mask = np.array(mask_img, dtype=bool)  # shape == gray.shape

        score, status = check_blurryiness(mask)
        detections.append([score, mask])
        
        # --- draw polygon ---
        #if DRAW: draw_polygon(poly_pts)
        
    to_infer = max(detections, key=lambda x: x[0])[1]

    # --- crop bounding box from mask ---

    ys, xs = np.where(to_infer)
    if len(xs) == 0:
        continue

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop = rgb[y1:y2, x1:x2]


    # --- infer species ---
    if ANIMAL_DETECTION:
        out = infer_species(crop)

        topk = 3
        # Get top 3 indices as plain Python ints
        topk_idx = np.argsort(out, axis=1)[:, -topk:][:, ::-1]

        species_preds = [(INAT_CLASSES[int(i)], float(out[0][int(i)])) for i in topk_idx[0]]

        # print result
        print("Detected animal:")
        print(f"Focus score  : {score:.1f} → {status}")
        print("Top 3 species predictions:")

        for name, conf in species_preds:
            print(f"  {name}: {conf * 100:.1f}%")

        print("-" * 40)
        
        # ---- Copy original raw file into top species folder ----
        top_species = species_preds[0][0]  # top-1 species
        species_folder = OUTPUT_FOLDER / top_species
        species_folder.mkdir(parents=True, exist_ok=True)  # create folder if needed

        # Copy the file
        dest_file = species_folder / file.name  # keep same name
        shutil.move(file, dest_file)


        


    #if DRAW: pil_img.show()


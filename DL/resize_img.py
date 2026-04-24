
import os
from PIL import Image
from pillow_heif import register_heif_opener   # ← HEIC support

register_heif_opener() 

# ====================== CONFIG ======================
INPUT_DIR = "raw_img"
OUTPUT_DIR = "dataset/assignment1"
TARGET_SIZE = (224, 224)
# ===================================================

def main():
    cwd = os.getcwd()
    input_full = os.path.join(cwd, INPUT_DIR)
    output_full = os.path.join(cwd, OUTPUT_DIR)
    
    print(f"Current dir     : {cwd}")
    print(f"Input folder    : {input_full}")
    print(f"Output folder   : {output_full}\n")
    
    if not os.path.exists(input_full):
        print(f"❌ Input folder not found: {input_full}")
        return
    
    os.makedirs(output_full, exist_ok=True)
    
    extensions = {'.jpg', '.jpeg', '.heic'}
    
    image_files = [f for f in os.listdir(input_full) 
                   if os.path.splitext(f.lower())[1] in extensions]
    
    if not image_files:
        print("❌ No images found.")
        return
    
    image_files.sort()
    
    print(f"Found {len(image_files)} images. Processing...\n")
    
    success = 0
    for i, filename in enumerate(image_files, 1):
        in_file = os.path.join(input_full, filename)
        out_file = os.path.join(output_full, f"img{i:04d}.jpg")
        
        try:
            with Image.open(in_file) as img:
                img = img.convert('RGB')
                resized = img.resize(TARGET_SIZE, Image.LANCZOS)
                resized.save(out_file, "JPEG", quality=95, optimize=True)
            
            print(f"✓ {filename:<40} → img{i:04d}.jpg")
            success += 1
        except Exception as e:
            print(f"✗ Failed {filename}: {e}")
    
    print("\n" + "="*70)
    print(f"✅ Done! {success}/{len(image_files)} images resized successfully.")
    print(f"   Saved to: {output_full}")
    print("="*70)

if __name__ == "__main__":
    main()
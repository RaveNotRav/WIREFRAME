import os
from PIL import Image, ImageDraw, ImageFont

def make_demo_images(output_dir="demo_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to use a default font, otherwise fallback to default PIL font
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except IOError:
        font = ImageFont.load_default()
        
    scenarios = [
        ("safe-pole.jpg", (50, 150, 50)),
        ("caution-pole.jpg", (150, 150, 50)),
        ("unsafe-pole.jpg", (200, 100, 50)),
        ("critical-pole.jpg", (200, 50, 50)),
        ("nowires.jpg", (100, 150, 200)),
        ("reject-blurry.jpg", (100, 100, 100)),
        ("nopole.jpg", (150, 100, 200)),
    ]
    
    for filename, color in scenarios:
        img = Image.new('RGB', (800, 600), color=color)
        d = ImageDraw.Draw(img)
        
        text = filename.split('.')[0].replace('-', ' ').upper()
        # center text roughly
        d.text((100, 250), text, fill=(255, 255, 255), font=font)
        
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, "JPEG")
        print(f"Created {filepath} - triggers {filename.split('.')[0]}")

if __name__ == "__main__":
    make_demo_images()

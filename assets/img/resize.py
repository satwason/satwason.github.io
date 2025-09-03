from PIL import Image
import os

# ===== CONFIGURATION =====
input_folder = r"C:\Users\Wason\Documents\GitHub\satwason.github.io\assets\img"
image_name = "combined_soil_air_temperature_analysis.png"  # Change this to the file you want to resize
output_folder = os.path.join(input_folder, "resized")
new_width = 1200   # Desired width (pixels)
new_height = 600  # Desired height (pixels)
keep_aspect_ratio = True  # Maintain original aspect ratio

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Full path of input and output files
img_path = os.path.join(input_folder, image_name)
save_path = os.path.join(output_folder, image_name)

# Open the image
img = Image.open(img_path)

# Resize to exact dimensions (ignores aspect ratio)
img = img.resize((new_width, new_height))


# Save resized image
img.save(save_path)
print(f"Resized image saved as: {save_path}")

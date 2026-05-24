from PIL import Image, ImageDraw, ImageFont
import os

# Create a white image
img = Image.new("RGB", (400, 200), color="white")
draw = ImageDraw.Draw(img)

# Write prescription text on it
draw.text((20, 20), "Patient: John Doe", fill="black")
draw.text((20, 50), "Rx: Warfarin 5mg daily", fill="black")
draw.text((20, 80), "Aspirin 100mg daily", fill="black")
draw.text((20, 110), "Omeprazole 20mg daily", fill="black")
draw.text((20, 140), "Dr. Smith", fill="black")

img.save("test_prescription.jpg")
print("Test image created successfully!")
import cv2
import numpy as np

# === CONFIGURATION ===
input_path = 'images/email.png'
output_path = 'new_email.png'
dark_threshold = 50
replacement_color = (215, 235, 250, 255)  # BGRA format (includes alpha)

# === LOAD IMAGE WITH ALPHA CHANNEL ===
image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

# === SPLIT CHANNELS ===
if image.shape[2] == 4:
    b, g, r, a = cv2.split(image)
    gray = cv2.cvtColor(cv2.merge((b, g, r)), cv2.COLOR_BGR2GRAY)
    mask = gray < dark_threshold
    # Replace only in BGR; preserve original alpha
    b[mask], g[mask], r[mask] = replacement_color[0], replacement_color[1], replacement_color[2]
    result = cv2.merge((b, g, r, a))
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < dark_threshold
    image[mask] = replacement_color[:3]
    result = image

# === SAVE RESULT ===
cv2.imwrite(output_path, result)

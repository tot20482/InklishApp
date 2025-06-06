import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

# Label map: 26 chữ thường + 26 chữ hoa
label_map =  [chr(i) for i in range(ord('A'), ord('Z')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)]

def binarize_image_pil(image_pil, threshold=100):
    """Chuyển ảnh PIL -> ảnh binary (ndarray)"""
    image = np.array(image_pil.convert('RGB'))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Dilation to thicken strokes
    kernel = np.ones((2, 2), np.uint8)
    thick = cv2.dilate(binary, kernel, iterations=1)
    return thick

def center_and_resize(image, size=(50, 50), padding=5):
    coords = cv2.findNonZero(image)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]
    padded = cv2.copyMakeBorder(cropped, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    h_, w_ = padded.shape
    scale = min(size[0] / w_, size[1] / h_)
    resized = cv2.resize(padded, (int(w_ * scale), int(h_ * scale)), interpolation=cv2.INTER_AREA)

    canvas = np.ones(size, dtype=np.uint8) * 0
    x_offset = (size[0] - resized.shape[1]) // 2
    y_offset = (size[1] - resized.shape[0]) // 2
    canvas[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized

    final = cv2.bitwise_not(canvas)
    return final

def preprocess_image(image_file):
    image_pil = Image.open(image_file).convert('RGB')
    binary = binarize_image_pil(image_pil)
    centered = center_and_resize(binary, size=(50, 50))

    # Chuyển ảnh về RGB 3 kênh
    rgb_image = np.stack([centered] * 3, axis=-1)

    image_array = rgb_image / 255.0
    return image_array.reshape(1, 50, 50, 3)

def recognize_text(image_file):
    img = preprocess_image(image_file)
    prediction = model.predict(img)
    index = np.argmax(prediction[0])
    if index < len(label_map):
        predicted_label = label_map[index]
    else:
        predicted_label = '?'  # fallback in case of out-of-bounds
    return {"text": predicted_label, "status": "ok"}



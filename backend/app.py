from flask import Flask, request, jsonify, render_template_string
from detect import detect_text
from recognize import recognize_text
from detect_and_recognize import detect_and_recognize
from PIL import Image

app = Flask(__name__)

# Trang upload ảnh đơn giản (dùng OCR luôn)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No image uploaded', 400
        image_file = request.files['image']
        result = detect_and_recognize(image_file.stream)
        return jsonify(result)

    html_form = """
    <!DOCTYPE html>
    <html>
    <head><title>Upload Image for OCR</title></head>
    <body>
        <h1>Upload Image for OCR</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Upload & Recognize</button>
        </form>
    </body>
    </html>
    """
    return render_template_string(html_form)

# API riêng cho detect
@app.route('/detect', methods=['POST'])
def detect_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    image = Image.open(image_file.stream)
    result = detect_text(image)
    return jsonify(result)

# API riêng cho recognize từ ảnh + list bbox
@app.route('/recognize', methods=['POST'])
def recognize_route():
    if 'image' not in request.files or not request.json or 'boxes' not in request.json:
        return jsonify({'error': 'Image and bounding boxes required'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream)
    boxes = request.json['boxes']

    results = []
    for box in boxes:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        char_img = image.crop((x1, y1, x2, y2))
        char_result = recognize_text(char_img)
        results.append({"box": box, "char": char_result["text"] if char_result["status"] == "ok" else None})

    return jsonify({'recognized_chars': results})

# API gộp detect + recognize (OCR toàn ảnh)
@app.route('/detect_and_recognize', methods=['POST'])
def ocr_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    result = detect_and_recognize(image_file.stream)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

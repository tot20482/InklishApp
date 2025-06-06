from PIL import Image
from io import BytesIO
from detect import detect_text
from recognize import recognize_text

def detect_and_recognize(image_file):
    image_pil = Image.open(image_file).convert('RGB')
    detection = detect_text(image_pil)
    boxes = detection["boxes"]

    if not boxes:
        return {"text": "", "characters": [], "status": "no characters found"}

    boxes = sorted(boxes, key=lambda b: b["xmin"])
    recognized = []

    for box in boxes:
        xmin, ymin = int(box["xmin"]), int(box["ymin"])
        xmax, ymax = int(box["xmax"]), int(box["ymax"])
        char_crop = image_pil.crop((xmin, ymin, xmax, ymax))

        buffer = BytesIO()
        char_crop.save(buffer, format="PNG")
        buffer.seek(0)

        result = recognize_text(buffer)
        if result["status"] == "ok":
            recognized.append({
                "char": result["text"],
                "box": box
            })

    text = ''.join([r["char"] for r in recognized])
    return {"text": text, "characters": recognized, "status": "ok"}

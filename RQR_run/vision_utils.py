import base64
from io import BytesIO

from PIL import Image


def encode_image_to_base64(image, max_side=1024):
    if max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        image = image.resize(
            (int(image.size[0] * scale), int(image.size[1] * scale)), Image.LANCZOS
        )
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def image_message(image_base64):
    return {"type": "image_url", "image_url": {"url": image_base64}}


def text_message(text):
    return {"type": "text", "text": text}

import base64
import io
import math

from PIL import Image


def estimate_tokens(text: str) -> int:
    """Rough token estimate used for runtime budget checks."""
    return len(text) // 4


def pil_to_base64_data_uri(pil_img, max_pixels: int = 2000000, min_pixels: int = 40000, quality: int = 85):
    """Convert a PIL image to a compressed JPEG data URI for model calls."""
    if pil_img.mode == 'RGBA':
        background = Image.new('RGB', pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    
    processed_img = process_image(pil_img, max_pixels=max_pixels, min_pixels=min_pixels)
    
    buffered = io.BytesIO()
    processed_img.save(buffered, format="JPEG", quality=quality)
    base64_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{base64_str}"


def process_image(image: Image.Image, max_pixels: int = 2000000, min_pixels: int = 40000, use_lanczos: bool = True) -> Image.Image:
    """Resize and normalize an image for model/tool inputs."""
    resample_method = Image.Resampling.LANCZOS if use_lanczos else Image.Resampling.NEAREST
    
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=resample_method)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=resample_method)

    if image.width < 28 or image.height < 28:
        resize_factor = 28 / min(image.width, image.height)
        width, height = int(image.width * resize_factor + 1), int(image.height * resize_factor + 1)
        image = image.resize((width, height), resample=resample_method)

    if image.width / image.height >= 200:
        width, height = image.width, int(image.width / 190 + 1)
        image = image.resize((width, height), resample=resample_method)

    if image.height / image.width >= 200:
        width, height = int(image.height / 190 + 1), image.height
        image = image.resize((width, height), resample=resample_method)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

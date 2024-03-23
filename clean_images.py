from PIL import Image, UnidentifiedImageError
import os


def check_image_channels(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                channels = len(img.getbands())
                if channels != 3:
                    img = img.convert("RGB")
                cleaned_path = os.path.join(output_folder, filename)
                img.save(cleaned_path)


def resize_image(final_size, image):
    size = image.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    resized_image = image.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(
        resized_image, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im


if __name__ == "__main__":
    input_folder = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\images"
    output_folder = r"C:\Users\nacho\New folder\AiCore\Facebook_Project\Cleaned_images"
    final_size = 1080

    check_image_channels(input_folder, output_folder)

    for n, item in enumerate(os.listdir(input_folder)[:5], 1):
        img_path = os.path.join(input_folder, item)
        try:
            with Image.open(img_path) as img:
                new_im = resize_image(final_size, img)
                output_path = os.path.join(output_folder, f"{n}_resized.jpg")
                new_im.save(output_path)
        except (IOError, OSError, ValueError, UnidentifiedImageError):
            print(f"Skipping non-image file: {img_path}")

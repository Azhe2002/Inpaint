import os
from PIL import Image


def resize_and_convert_images(directory):
    """
    Resize all .jpg and .png images in the directory to 128x128 pixels,
    convert them to .jpg format, and overwrite the original files.

    :param directory: Path to the directory containing the images.
    """
    if not os.path.isdir(directory):
        print(f"The provided path '{directory}' is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip if it's not a file
        if not os.path.isfile(file_path):
            continue

        # Process only .jpg and .png files
        if filename.lower().endswith(('.jpg', '.png')):
            try:
                with Image.open(file_path) as img:
                    # Resize the image to 128x128 using LANCZOS resampling
                    img = img.resize((128, 128), Image.Resampling.LANCZOS)

                    # Convert to RGB (necessary for saving as .jpg)
                    img = img.convert("RGB")

                    # Handle PNG files: rename to .jpg after conversion
                    if filename.lower().endswith('.png'):
                        new_file_path = os.path.splitext(file_path)[0] + ".jpg"
                        os.remove(file_path)  # Remove the original PNG file
                    else:
                        new_file_path = file_path  # Keep the same path for JPG files

                    # Save the image back as .jpg
                    img.save(new_file_path, "JPEG")
                    print(f"Processed and saved: {new_file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")


if __name__ == "__main__":
    directory = input("Enter the path to the directory containing the images: ")
    resize_and_convert_images(directory)
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim


class ImageProcessor:
    def __init__(self, path=None, new_width=None, new_height=None, mode='RGB'):
        self.path = path

        self.arr = None
        self.new_width = new_width
        self.new_height = new_height
        self.mode = mode

        if path:
            self.load()
        else:
            print('No image provided')

    def load(self):
        """
        Loads an image and resizes it while maintaining aspect ratio. Can load in color (RGB) or grayscale (L).

        Returns:
        - img_array: A numpy array of the resized image.
        """
        with Image.open(self.path) as img:
            # Maintain aspect ratio
            aspect_ratio = img.height / img.width
            self.new_height = int(self.new_width * aspect_ratio)

            # Resize image
            img_resized = img.resize((self.new_width, self.new_height))

            # Convert to the specified mode if necessary
            if img_resized.mode != self.mode:
                img_resized = img_resized.convert(self.mode)

            self.arr = np.asarray(img_resized)

    def save(self, output_path, image_arr=None):
        """Saves the current or provided numpy array as an image."""
        if image_arr is None:
            image_arr = self.arr
        if image_arr is not None:
            img = Image.fromarray(image_arr.astype(np.uint8), mode=self.mode)
            img.save(output_path)
        else:
            print("Error: No image array to save.")


def calculate_similarity(image1_array, image2_array):
    """Calculates and returns the structural similarity index between two images."""
    if image1_array.shape == image2_array.shape:
        return ssim(image1_array, image2_array, multichannel=True)
    else:
        print("Error: Images must have the same dimensions.")
        return None

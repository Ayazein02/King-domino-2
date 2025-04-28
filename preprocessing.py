import cv2 as cv
import numpy as np
import glob
from pathlib import Path
from icecream import ic
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from scipy.stats import skew, kurtosis, entropy
    SCIPY_AVAILABLE = True
except ImportError:
    ic("Warning: SciPy not installed. Skew, kurtosis, and entropy will be skipped.")
    SCIPY_AVAILABLE = False

@dataclass
class PreprocessorConfig:
    image_dir: str
    image_pattern: str
    output_dir: str
    target_size: Tuple[int, int] = (500, 500)
    grid_size: Tuple[int, int] = (5, 5)
    save_images: bool = True

@dataclass
class ImageProcessorConfig:
    target_size: Tuple[int, int] = (500, 500)
    blur_threshold: float = 100.0
    tile_size: int = 64
    nlm_h: int = 10
    nlm_template_window: int = 7
    nlm_search_window: int = 21
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75
    clahe_clip_limit_low: float = 3.5  # Adjusted for better contrast
    clahe_clip_limit_high: float = 2.0
    brightness_low: float = 90.0
    brightness_high: float = 150.0

class DirectoryManager:
    @staticmethod
    def ensure_output_dir(output_dir: str) -> str:
        """
        Ensure the output directory exists.

        Args:
            output_dir (str): Directory path.

        Returns:
            str: Output directory path.
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            ic(f"Ensured output directory: {output_dir}")
            return output_dir
        except Exception as e:
            ic(f"Error creating output directory {output_dir}: {e}")
            raise

class ImageProcessor:
    def __init__(self, config: ImageProcessorConfig):
        """
        Initialize the image processor with configuration.

        Args:
            config (ImageProcessorConfig): Configuration for image processing.
        """
        self.config = config

    def load_and_resize(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and resize an image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            np.ndarray: Resized image, or None if loading fails.
        """
        try:
            img = cv.imread(image_path)
            if img is None:
                ic(f"Error: Could not load image at {image_path}")
                return None
            img_resized = cv.resize(img, self.config.target_size, interpolation=cv.INTER_AREA)
            ic(f"Image resized to {self.config.target_size}")
            return img_resized
        except Exception as e:
            ic(f"Error loading/resizing image {image_path}: {e}")
            return None

    def denoise(self, image: np.ndarray, method: str = 'auto') -> Optional[np.ndarray]:
        """
        Denoise an image using NLM or bilateral filtering based on brightness and blur.

        Args:
            image (np.ndarray): Input BGR image.
            method (str): Denoising method ('auto', 'nlm', 'bilateral').

        Returns:
            np.ndarray: Denoised image, or None on failure.
        """
        try:
            if image is None or image.size == 0:
                ic("Error: Invalid image for denoising.")
                return None

            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            brightness = np.median(hsv[:, :, 2])
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            blur_metric = cv.Laplacian(gray, cv.CV_64F).var()

            if blur_metric > self.config.blur_threshold:
                ic(f"Skipping denoising: high sharpness (blur metric: {blur_metric})")
                return image

            method = 'nlm' if method == 'auto' and brightness < 100 else 'bilateral'
            ic(f"Denoising method: {method}, brightness: {brightness:.2f}, blur: {blur_metric:.2f}")

            if method == 'nlm':
                denoised = cv.fastNlMeansDenoisingColored(
                    image, None, self.config.nlm_h, self.config.nlm_h,
                    self.config.nlm_template_window, self.config.nlm_search_window
                )
            else:
                denoised = cv.bilateralFilter(
                    image, self.config.bilateral_d,
                    self.config.bilateral_sigma_color, self.config.bilateral_sigma_space
                )
            return denoised
        except Exception as e:
            ic(f"Error denoising image: {e}")
            return None

    def standardize_lighting(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Standardize lighting by adjusting brightness in HSV space.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Lighting-standardized image, or None on failure.
        """
        try:
            if image is None or image.size == 0:
                ic("Error: Invalid image for lighting standardization.")
                return None

            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv)
            brightness = np.median(v)
            ic(f"Median brightness: {brightness:.2f}")

            if brightness > self.config.brightness_high:
                v = np.clip(v * 0.9, 0, 255).astype(np.uint8)  # Darken bright images
                ic("Darkening bright image")
            elif brightness < self.config.brightness_low:
                clahe = cv.createCLAHE(clipLimit=self.config.clahe_clip_limit_low, tileGridSize=(6, 6))
                v = clahe.apply(v)
                ic("Applying high-contrast CLAHE for low brightness")
            else:
                clahe = cv.createCLAHE(clipLimit=self.config.clahe_clip_limit_high, tileGridSize=(6, 6))
                v = clahe.apply(v)
                ic("Applying standard CLAHE")

            return cv.cvtColor(cv.merge([h, s, v]), cv.COLOR_HSV2BGR)
        except Exception as e:
            ic(f"Error standardizing lighting: {e}")
            return None

    def sharpen(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Sharpen the image using tile-based sharpening.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Sharpened image, or None on failure.
        """
        try:
            if image is None or image.size == 0:
                ic("Error: Invalid image for sharpening.")
                return None

            h, w = image.shape[:2]
            result = image.copy()
            ic(f"Sharpening image with shape: {image.shape}")

            for y in range(0, h, self.config.tile_size):
                for x in range(0, w, self.config.tile_size):
                    tile = result[y:y + self.config.tile_size, x:x + self.config.tile_size]
                    if tile.size == 0:
                        continue
                    gray = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
                    blur_metric = cv.Laplacian(gray, cv.CV_64F).var()

                    if blur_metric < self.config.blur_threshold:
                        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        tile = cv.filter2D(tile, -1, kernel)
                    else:
                        gaussian = cv.GaussianBlur(tile, (0, 0), 3.0)
                        tile = cv.addWeighted(tile, 1.5, gaussian, -0.5, 0)
                    result[y:y + self.config.tile_size, x:x + self.config.tile_size] = tile
            ic(f"Image sharpened with shape: {result.shape}")
            return result
        except Exception as e:
            ic(f"Error sharpening image: {e}")
            return None

    def normalize(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize the image in LAB color space.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Normalized image, or None on failure.
        """
        try:
            if image is None or image.size == 0:
                ic("Error: Invalid image for normalization.")
                return None

            lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab)
            l_norm = cv.normalize(l, None, 0, 255, cv.NORM_MINMAX)
            normalized = cv.cvtColor(cv.merge([l_norm, a, b]), cv.COLOR_LAB2BGR)
            ic(f"Image normalized with shape: {normalized.shape}")
            return normalized
        except Exception as e:
            ic(f"Error normalizing image: {e}")
            return None

    def process(self, image_path: str, output_dir: str, save_image: bool = True) -> bool:
        """
        Process an image through denoising, lighting standardization, sharpening, and normalization.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save preprocessed image.
            save_image (bool): Whether to save the preprocessed image.

        Returns:
            bool: True if processing succeeds, False otherwise.
        """
        filename = Path(image_path).name
        output_path = Path(output_dir) / f"preprocessed_{filename}"

        if save_image and output_path.exists():
            ic(f"Skipping existing file: {output_path}")
            return True

        img = self.load_and_resize(image_path)
        if img is None:
            return False

        try:
            img = self.denoise(img)
            if img is None:
                return False
            img = self.standardize_lighting(img)
            if img is None:
                return False
            img = self.sharpen(img)
            if img is None:
                return False
            img = self.normalize(img)
            if img is None:
                return False

            if save_image:
                if cv.imwrite(str(output_path), img):
                    ic(f"Saved preprocessed image: {output_path}")
                else:
                    ic(f"Failed to save preprocessed image: {output_path}")
                    return False
            return True
        except Exception as e:
            ic(f"Error processing {filename}: {e}")
            return False

class Preprocessor:
    def __init__(self, config: PreprocessorConfig):
        """
        Initialize the preprocessor with configuration.

        Args:
            config (PreprocessorConfig): Configuration for preprocessing.
        """
        self.config = config
        self.processor = ImageProcessor(ImageProcessorConfig(target_size=config.target_size))

    def run(self) -> bool:
        """
        Run preprocessing on all images in the input directory.

        Returns:
            bool: True if at least one image is processed successfully, False otherwise.
        """
        try:
            output_dir = DirectoryManager.ensure_output_dir(self.config.output_dir)
            image_paths = sorted(glob.glob(str(Path(self.config.image_dir) / self.config.image_pattern)))

            if not image_paths:
                ic(f"No images found in {self.config.image_dir} matching {self.config.image_pattern}")
                return False

            success_count = sum(
                self.processor.process(path, output_dir, self.config.save_images)
                for path in image_paths
            )
            ic(f"Processed {success_count}/{len(image_paths)} images successfully.")
            return success_count > 0
        except Exception as e:
            ic(f"Error running preprocessor: {e}")
            return False

if __name__ == "__main__":
    config = PreprocessorConfig(
        image_dir=r'C:\Users\stign\Desktop\King Domino-20250410T120039Z-001\King Domino dataset\Cropped and perspective corrected boards',
        image_pattern='*.jpg',
        output_dir=r'C:\Users\stign\Desktop\King Domino-20250410T120039Z-001\Preprocessed_Images'
    )
    preprocessor = Preprocessor(config)
    preprocessor.run()
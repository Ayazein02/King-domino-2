import cv2 as cv
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class ImageProcessorConfig:
    grid_rows: int = 5
    grid_cols: int = 5
    target_img_width: int = 500
    target_img_height: int = 500
    hog_win_size: tuple[int, int] = (96, 96)
    hog_block_size: tuple[int, int] = (16, 16)
    hog_block_stride: tuple[int, int] = (8, 8)
    hog_cell_size: tuple[int, int] = (8, 8)
    hog_nbins: int = 9
    hsv_thresholds: dict = field(default_factory=lambda: {
        "lake": ([94, 225, 132], [119, 255, 210]),
        "forest": ([20, 78, 40], [80, 220, 113]),
        "grassland": ([25, 190, 113], [51, 255, 195]),
        "field": ([13, 220, 178], [37, 255, 239]),
        "swamp": ([10, 20, 50], [30, 200, 180]),
        "table": ([15, 112, 128], [25, 203, 200]),
        "unknown": ([0, 0, 0], [255, 255, 255]),
    })
    fallback_terrain: str = "unknown"

class ImageProcessor:
    def __init__(self, config: ImageProcessorConfig):
        self.config = config
        self.hog = cv.HOGDescriptor(
            config.hog_win_size,
            config.hog_block_size,
            config.hog_block_stride,
            config.hog_cell_size,
            config.hog_nbins
        )

    def load_and_preprocess(self, image_path: str):
        img = cv.imread(image_path)
        if img is None:
            return None, None, None
        resized = cv.resize(img, (self.config.target_img_width, self.config.target_img_height))
        hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)
        return resized, img, hsv

    def extract_tiles(self, image: np.ndarray, rows: int, cols: int):
        h, w = image.shape[:2]
        tile_h, tile_w = h // rows, w // cols
        tiles = []
        for y in range(rows):
            row = []
            for x in range(cols):
                tile = image[y*tile_h:(y+1)*tile_h, x*tile_w:(x+1)*tile_w]
                row.append(tile)
            tiles.append(row)
        return tiles

    def extract_tiles_from_path(self, image_path: str) -> Optional[List[List[np.ndarray]]]:
        image = cv.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        resized = cv.resize(image, (self.config.target_img_width, self.config.target_img_height))
        tiles = self.extract_tiles(resized, self.config.grid_rows, self.config.grid_cols)
        return tiles

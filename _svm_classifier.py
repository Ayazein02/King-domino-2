import cv2 as cv
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from icecream import ic
from typing import Optional, List, Tuple

class SVMClassifier:
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        hog_win_size: tuple[int, int] = (96, 96),
        hog_block_size: tuple[int, int] = (16, 16),
        hog_block_stride: tuple[int, int] = (8, 8),
        hog_cell_size: tuple[int, int] = (8, 8),
        hog_nbins: int = 9
    ):
        """
        Initialize the SVM classifier with HOG descriptor and load model/scaler.

        Args:
            model_path (str): Path to the SVM model file.
            scaler_path (str): Path to the scaler file.
            hog_win_size, hog_block_size, hog_block_stride, hog_cell_size, hog_nbins: HOG parameters.
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.hog_win_size = hog_win_size
        self.hog_block_size = hog_block_size
        self.hog_block_stride = hog_block_stride
        self.hog_cell_size = hog_cell_size
        self.hog_nbins = hog_nbins

        try:
            self.hog = cv.HOGDescriptor(
                hog_win_size, hog_block_size, hog_block_stride, hog_cell_size, hog_nbins
            )
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            ic(f"SVM model and scaler loaded from {model_path} and {scaler_path}.")
        except Exception as e:
            ic(f"Error loading SVM model or scaler: {e}")
            raise

    def retrain(self, features: np.ndarray, labels: np.ndarray, save: bool = True) -> None:
        """
        Retrain the SVM model with new features and labels.

        Args:
            features (np.ndarray): Training feature vectors.
            labels (np.ndarray): Corresponding labels.
            save (bool): Whether to save the retrained model and scaler.
        """
        try:
            ic(f"Retraining SVM with {features.shape[0]} samples, feature length: {features.shape[1]}")
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features)
            self.model = SVC(kernel='rbf', C=1.0, probability=True)
            self.model.fit(scaled_features, labels)
            if save:
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                ic(f"SVM model and scaler saved to {self.model_path} and {self.scaler_path}")
        except Exception as e:
            ic(f"Error retraining SVM model: {e}")
            raise

    def compute_hog_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute HOG features for an image.

        Args:
            image (np.ndarray): Input image (grayscale or BGR).

        Returns:
            np.ndarray: HOG feature vector, or None on failure.
        """
        try:
            if image is None or image.size == 0:
                ic("Error: Invalid image for HOG computation.")
                return None
            if len(image.shape) == 3:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            if image.shape[:2] != self.hog_win_size:
                image = cv.resize(image, self.hog_win_size, interpolation=cv.INTER_AREA)
            features = self.hog.compute(image)
            if features is None or len(features) == 0:
                ic("Error: HOG computation returned empty features.")
                return None
            ic(f"Computed HOG features with length: {len(features.flatten())}")
            return features.flatten()
        except Exception as e:
            ic(f"Error computing HOG features: {e}")
            return None

    def preprocess_tile(self, tile: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess a tile for SVM prediction by resizing and converting to grayscale.

        Args:
            tile (np.ndarray): Input BGR tile image.

        Returns:
            Grayscale tile image, or None on failure.
        """
        try:
            if tile is None or tile.size == 0:
                ic("Error: Invalid tile (None or empty) for preprocessing.")
                return None
            tile_resized = cv.resize(tile, self.hog_win_size, interpolation=cv.INTER_AREA)
            gray_tile = cv.cvtColor(tile_resized, cv.COLOR_BGR2GRAY)
            ic(f"Preprocessed tile with shape: {gray_tile.shape}")
            return gray_tile
        except Exception as e:
            ic(f"Error preprocessing tile: {e}")
            return None

    def predict_tile(self, gray_tile: Optional[np.ndarray]) -> str:
        """
        Predict the terrain type of a tile using the SVM model.

        Args:
            gray_tile (np.ndarray): Grayscale tile image.

        Returns:
            Predicted terrain label (lowercase), or "svm_error" if prediction fails.
        """
        try:
            if gray_tile is None or gray_tile.size == 0:
                ic("Error: Invalid gray tile (None or empty) for prediction.")
                return "svm_error"

            features = self.compute_hog_features(gray_tile)
            if features is None:
                ic("Error: HOG features are None.")
                return "svm_error"

            features = features.reshape(1, -1)
            ic(f"HOG feature length for prediction: {features.shape[1]}")
            scaled_features = self.scaler.transform(features)
            prediction = self.model.predict(scaled_features)[0]
            ic(f"SVM prediction: {prediction}")
            return str(prediction).lower()
        except ValueError as ve:
            ic(f"ValueError in SVM prediction: {ve}")
            return "svm_error"
        except AttributeError as ae:
            ic(f"AttributeError in SVM prediction (model/scaler issue): {ae}")
            return "svm_error"
        except Exception as e:
            ic(f"Unexpected error in SVM prediction: {e}")
            return "svm_error"

    def collect_training_data(self, tiles: List[np.ndarray], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect HOG features and labels for training.

        Args:
            tiles (List[np.ndarray]): List of tile images.
            labels (List[str]): Corresponding terrain labels.

        Returns:
            Tuple of feature matrix and label array.
        """
        features_list = []
        valid_labels = []
        for tile, label in zip(tiles, labels):
            gray_tile = self.preprocess_tile(tile)
            if gray_tile is None:
                continue
            features = self.compute_hog_features(gray_tile)
            if features is not None:
                features_list.append(features)
                valid_labels.append(label.lower())
        if not features_list:
            raise ValueError("No valid HOG features collected for training.")
        return np.array(features_list), np.array(valid_labels)
import os
import time
import glob
import cv2
import cv2 as cv
import numpy as np
from typing import Dict, List, Tuple, Optional
from icecream import ic
from dataclasses import dataclass, field
from preprocessing import Preprocessor, PreprocessorConfig
from _svm_classifier import SVMClassifier
from crown_detection7 import CrownDetector
from _visualizer import Visualizer
from _dataexport import DataExporter
from tile_classification import ImageProcessor, ImageProcessorConfig


# === REPLACEMENT for scoring ===
def compute_score(terrain_grid, crown_grid):
    visited = np.zeros_like(terrain_grid, dtype=bool)
    total_score = 0
    rows, cols = terrain_grid.shape

    def flood_fill(r, c, terrain_type):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            visited[r, c] or terrain_grid[r, c] != terrain_type):
            return 0, 0
        visited[r, c] = True
        area_size = 1
        crowns = crown_grid[r, c]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            add_area, add_crowns = flood_fill(r + dr, c + dc, terrain_type)
            area_size += add_area
            crowns += add_crowns
        return area_size, crowns

    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and terrain_grid[r, c] != 'table':
                area_size, crowns = flood_fill(r, c, terrain_grid[r, c])
                total_score += area_size * crowns
    return total_score

def find_preprocessed_images(folder, pattern='preprocessed_*.jpg'):
    return sorted(glob.glob(os.path.join(folder, pattern)))

def main():
    start_time = time.time()

    import pandas as pd

    # --- Load Ground Truth Data ---
    ground_truth_tile = pd.read_csv(r"C:\Users\ayaal\OneDrive\Skrivebord\DAKI 24 1\GT's\ground_truth.csv")
    ground_truth_tile['filename'] = ground_truth_tile['filename'].str.strip()

    ground_truth_score = pd.read_excel(r"C:\Users\ayaal\OneDrive\Skrivebord\DAKI 24 1\GT's\Ground_truth_score.xlsx")
    ground_truth_score['filename'] = ground_truth_score['filename'].str.strip()

    tile_correct = 0
    tile_total = 0
    score_differences = []


    # --- 1. Preprocessing ---
    preprocessor = Preprocessor(PreprocessorConfig(
    image_dir=r"C:\Users\ayaal\OneDrive\Skrivebord\Design og udvikling af AI-systemer\King Domino dataset\Cropped and perspective corrected boards",
    image_pattern="*.jpg",
    output_dir=r"C:\Users\ayaal\OneDrive\Skrivebord\DAKI 24 1\Status_22-04-2025\Preprocessed_Images_new"
))

    preprocessor.run()

    # --- 2. Tile Split + Terrain Classification ---
    image_processor = ImageProcessor(ImageProcessorConfig())
    svm_classifier = SVMClassifier(
    model_path=r'C:\Users\ayaal\OneDrive\Skrivebord\DAKI 24 1\_svm_model_z.joblib',
    scaler_path=r'C:\Users\ayaal\OneDrive\Skrivebord\DAKI 24 1\_scaler_z.joblib'
)

    # --- 3. Crown Detection ---
    crown_detector = CrownDetector(
        template_folder=r'C:\Users\ayaal\OneDrive\Skrivebord\DAKI 24 1\crown_templates copy 1'
    )

    # --- 4. Visualizer + Data Exporter ---
    visualizer = Visualizer(tile_width=100, tile_height=100)
    data_exporter = DataExporter(output_dir=r'C:\Users\ayaal\OneDrive\Skrivebord\DAKI 24 1\Status_22-04-2025\Preprocessed_Images_new\data_export')

    # --- Find preprocessed images ---
    preprocessed_folder = r'C:\Users\ayaal\OneDrive\Skrivebord\DAKI 24 1\Status_22-04-2025\Preprocessed_Images_new'
    image_paths = find_preprocessed_images(preprocessed_folder)

    all_results = []

    for img_path in image_paths:
        print(f'Processing: {os.path.basename(img_path)}')

        # --- Tile Split ---
        bgr_tiles = image_processor.extract_tiles_from_path(img_path)
        if not bgr_tiles:
            print(f"Skipping {img_path} (tile split failed)")
            continue

        terrain_grid = np.full((5, 5), 'unknown', dtype=object)
        crown_grid = np.zeros((5, 5), dtype=int)

        for row_idx, row_tiles in enumerate(bgr_tiles):
            for col_idx, tile in enumerate(row_tiles):
                if tile is None:
                    continue

                # --- Terrain Classification ---
                gray_tile = svm_classifier.preprocess_tile(tile)
                terrain = svm_classifier.predict_tile(gray_tile)
                terrain_grid[row_idx, col_idx] = terrain.lower()

                # --- Crown Detection ---
                crown_count, _, _ = crown_detector.detect_crowns(tile)
                crown_grid[row_idx, col_idx] = crown_count

        # --- Compute Score ---
        score = compute_score(terrain_grid, crown_grid)

        # --- Tile Classification Evaluation ---
        current_filename = os.path.basename(img_path).replace("preprocessed_", "")
        gt_tiles = ground_truth_tile[ground_truth_tile['filename'] == current_filename]

        if not gt_tiles.empty:
            for index, gt_row in gt_tiles.iterrows():
                tile_id = int(gt_row['tile_id'])
                gt_label = str(gt_row['label']).lower()

                row = (tile_id - 1) // 5
                col = (tile_id - 1) % 5
                pred_label = terrain_grid[row, col]

                if pred_label == gt_label:
                    tile_correct += 1
                tile_total += 1
        else:
            ic(f"⚠️ Ingen tile-GT fundet for {current_filename}")

        # --- Score Evaluation ---
        gt_score_row = ground_truth_score[ground_truth_score['filename'] == current_filename]

        if not gt_score_row.empty:
            gt_score = int(gt_score_row['score'].values[0])
            score_diff = abs(score - gt_score)
            score_differences.append(score_diff)
            ic(f"✅ Score for {current_filename}: Beregnet={score}, GT={gt_score}, Diff={score_diff}")
        else:
            ic(f"⚠️ Ingen GT-score fundet for {current_filename}")


        # --- Save Annotated Visualization ---
        vis_output_dir = os.path.join(preprocessed_folder, 'visualizations')
        os.makedirs(vis_output_dir, exist_ok=True)
        annotated_image_path = os.path.join(vis_output_dir, f"annotated_{os.path.basename(img_path)}")

        # Konverter terrain_grid (strings) til analysis_results (list of dicts)
        analysis_results = []
        for row in terrain_grid:
            analysis_row = []
            for terrain in row:
                analysis_row.append({
                    'terrain_hsv': terrain,
                    'terrain_svm': terrain,
                    'terrain_final': terrain
                })
            analysis_results.append(analysis_row)

        # Konverter crown_grid til analysis_results (list of dicts)

# --- Indlæs original billede --- 
        original_image_path = os.path.join(
            r"C:\Users\ayaal\OneDrive\Skrivebord\Design og udvikling af AI-systemer\King Domino dataset\Cropped and perspective corrected boards",
            os.path.basename(img_path).replace("preprocessed_", "")
        )
        original_image = cv2.imread(original_image_path)

        # --- Lav annotated image ---
        annotated_img = visualizer.create_annotated_tile_grid(
            bgr_tiles,
            analysis_results,
            original_image,         # <- her bruger vi rigtigt original
            cv2.imread(img_path)     # <- preprocessed billede
        )



        if annotated_img is not None:
            cv2.putText(annotated_img, f"Score: {score} pts", (10, annotated_img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imwrite(annotated_image_path, annotated_img)

        # --- Collect Results ---
        all_results.append({
            'filename': os.path.basename(img_path),
            'score': score,
            'crowns_found': np.sum(crown_grid)
        })

    # --- Save results to CSV ---
    if all_results:
        data_exporter.export_results(all_results)

    # --- Final Evaluation Metrics ---

# Tile classification accuracy
    if tile_total > 0:
        tile_accuracy = tile_correct / tile_total * 100
        print(f"\n Tile Classification Accuracy: {tile_accuracy:.2f}% ({tile_correct}/{tile_total})")
    else:
        print("\n Ingen tile evaluation mulig (tile_total=0)")

    # Score difference evaluation
    if score_differences:
        avg_score_diff = np.mean(score_differences)
        print(f" Average Score Difference: {avg_score_diff:.2f} points")
    else:
        print("\n Ingen score evaluation mulig (ingen GT scores fundet)")


    print(f" Finished processing {len(image_paths)} images in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

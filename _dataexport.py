import os
import csv
import numpy as np
from typing import Dict, List, Any
from icecream import ic

class DataExporter:
    def __init__(self, output_dir: str):
        """
        Initialize the data exporter with output directory.

        Args:
            output_dir (str): Directory to save exported data.
        """
        self.output_dir = output_dir
        ic(f"DataExporter initialized with output_dir: {self.output_dir}")

    def export_hsv_data_to_csv(self, tile_data: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Export HSV-related tile data to a CSV file.

        Args:
            tile_data (List[Dict[str, Any]]): List of tile data dictionaries.
            output_path (str): Path to save the CSV file.

        Returns:
            bool: True if export succeeds, False otherwise.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fieldnames = [
                'image_filename', 'tile_num', 'row', 'col',
                'terrain_hsv', 'terrain_svm', 'terrain_final',
                'gt_terrain', 'is_correct_hsv', 'is_correct_svm', 'is_correct_final',
                'h_median', 's_median', 'v_median', 'h_mean', 's_mean', 'v_mean',
                'crown_count', 'crown_scores'
            ]
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for tile in tile_data:
                    hsv_stats = tile.get('hsv_stats', {'median': (0, 0, 0), 'mean': (0, 0, 0)})
                    row = {
                        'image_filename': tile.get('image_filename', ''),
                        'tile_num': tile.get('tile_num', 0),
                        'row': tile.get('row', 0),
                        'col': tile.get('col', 0),
                        'terrain_hsv': tile.get('terrain_hsv', 'unknown'),
                        'terrain_svm': tile.get('terrain_svm', 'unknown'),
                        'terrain_final': tile.get('terrain_final', 'unknown'),
                        'gt_terrain': tile.get('gt_terrain', 'unknown'),
                        'is_correct_hsv': tile.get('is_correct_hsv', 0),
                        'is_correct_svm': tile.get('is_correct_svm', 0),
                        'is_correct_final': tile.get('is_correct_final', 0),
                        'h_median': hsv_stats['median'][0] if hsv_stats['median'][0] is not None else 0,
                        's_median': hsv_stats['median'][1] if hsv_stats['median'][1] is not None else 0,
                        'v_median': hsv_stats['median'][2] if hsv_stats['median'][2] is not None else 0,
                        'h_mean': hsv_stats['mean'][0] if hsv_stats['mean'][0] is not None else 0,
                        's_mean': hsv_stats['mean'][1] if hsv_stats['mean'][1] is not None else 0,
                        'v_mean': hsv_stats['mean'][2] if hsv_stats['mean'][2] is not None else 0,
                        'crown_count': tile.get('crown_count', 0),
                        'crown_scores': ','.join(map(str, tile.get('crown_scores', [])))
                    }
                    writer.writerow(row)
            ic(f"Exported HSV data to {output_path}")
            return True
        except Exception as e:
            ic(f"Error exporting HSV data to {output_path}: {e}")
            return False

    def export_hog_data_to_csv(self, tile_data: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Export HOG feature data to a CSV file.

        Args:
            tile_data (List[Dict[str, Any]]): List of tile data dictionaries.
            output_path (str): Path to save the CSV file.

        Returns:
            bool: True if export succeeds, False otherwise.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_filename', 'tile_num', 'row', 'col', 'terrain_final', 'hog_features'])
                for tile in tile_data:
                    hog = tile.get('hog_features', [])
                    writer.writerow([
                        tile.get('image_filename', ''),
                        tile.get('tile_num', 0),
                        tile.get('row', 0),
                        tile.get('col', 0),
                        tile.get('terrain_final', 'unknown'),
                        ','.join(map(str, hog)) if hog else ''
                    ])
            ic(f"Exported HOG data to {output_path}")
            return True
        except Exception as e:
            ic(f"Error exporting HOG data to {output_path}: {e}")
            return False

    def calculate_hsv_statistics_by_terrain(self, tile_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate HSV statistics grouped by final terrain type.

        Args:
            tile_data (List[Dict[str, Any]]): List of tile data dictionaries.

        Returns:
            Dict mapping terrain types to statistics.
        """
        stats = {}
        for tile in tile_data:
            terrain_final = tile.get('terrain_final', 'unknown')
            hsv_stats = tile.get('hsv_stats', {'median': (0, 0, 0), 'mean': (0, 0, 0)})

            if terrain_final not in stats:
                stats[terrain_final] = {
                    'h_median': [], 's_median': [], 'v_median': [],
                    'h_mean': [], 's_mean': [], 'v_mean': [],
                    'gt_matches_hsv': [], 'gt_matches_svm': [], 'gt_matches_final': [],
                    'crown_counts': [], 'total_predictions': 0
                }

            stats[terrain_final]['h_median'].append(hsv_stats['median'][0] if hsv_stats['median'][0] is not None else 0)
            stats[terrain_final]['s_median'].append(hsv_stats['median'][1] if hsv_stats['median'][1] is not None else 0)
            stats[terrain_final]['v_median'].append(hsv_stats['median'][2] if hsv_stats['median'][2] is not None else 0)
            stats[terrain_final]['h_mean'].append(hsv_stats['mean'][0] if hsv_stats['mean'][0] is not None else 0)
            stats[terrain_final]['s_mean'].append(hsv_stats['mean'][1] if hsv_stats['mean'][1] is not None else 0)
            stats[terrain_final]['v_mean'].append(hsv_stats['mean'][2] if hsv_stats['mean'][2] is not None else 0)
            stats[terrain_final]['gt_matches_hsv'].append(tile.get('is_correct_hsv', 0))
            stats[terrain_final]['gt_matches_svm'].append(tile.get('is_correct_svm', 0))
            stats[terrain_final]['gt_matches_final'].append(tile.get('is_correct_final', 0))
            stats[terrain_final]['crown_counts'].append(tile.get('crown_count', 0))
            stats[terrain_final]['total_predictions'] += 1

        return stats

    def export_hsv_statistics_to_csv(self, stats: Dict[str, Dict[str, Any]], output_path: str) -> bool:
        """
        Export HSV statistics to a CSV file.

        Args:
            stats (Dict[str, Dict[str, Any]]): Statistics dictionary.
            output_path (str): Path to save the CSV file.

        Returns:
            bool: True if export succeeds, False otherwise.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'terrain', 'h_median_avg', 's_median_avg', 'v_median_avg',
                    'h_mean_avg', 's_mean_avg', 'v_mean_avg',
                    'accuracy_hsv', 'accuracy_svm', 'accuracy_final',
                    'avg_crown_count', 'total_predictions'
                ])
                for terrain, data in stats.items():
                    total = data['total_predictions']
                    row = [
                        terrain,
                        np.mean(data['h_median']) if data['h_median'] else 0,
                        np.mean(data['s_median']) if data['s_median'] else 0,
                        np.mean(data['v_median']) if data['v_median'] else 0,
                        np.mean(data['h_mean']) if data['h_mean'] else 0,
                        np.mean(data['s_mean']) if data['s_mean'] else 0,
                        np.mean(data['v_mean']) if data['v_mean'] else 0,
                        sum(data['gt_matches_hsv']) / total if total > 0 else 0,
                        sum(data['gt_matches_svm']) / total if total > 0 else 0,
                        sum(data['gt_matches_final']) / total if total > 0 else 0,
                        np.mean(data['crown_counts']) if data['crown_counts'] else 0,
                        total
                    ]
                    writer.writerow(row)
            ic(f"Exported HSV statistics to {output_path}")
            return True
        except Exception as e:
            ic(f"Error exporting HSV statistics to {output_path}: {e}")
            return False

    def print_hsv_statistics(self, stats: Dict[str, Dict[str, Any]]) -> None:
        """
        Print HSV statistics for each terrain type.

        Args:
            stats (Dict[str, Dict[str, Any]]): Statistics dictionary.
        """
        for terrain, data in stats.items():
            total = data['total_predictions']
            ic(f"Statistics for {terrain}:")
            ic(f"  H Median Avg: {np.mean(data['h_median']) if data['h_median'] else 0:.2f}")
            ic(f"  S Median Avg: {np.mean(data['s_median']) if data['s_median'] else 0:.2f}")
            ic(f"  V Median Avg: {np.mean(data['v_median']) if data['v_median'] else 0:.2f}")
            ic(f"  H Mean Avg: {np.mean(data['h_mean']) if data['h_mean'] else 0:.2f}")
            ic(f"  S Mean Avg: {np.mean(data['s_mean']) if data['s_mean'] else 0:.2f}")
            ic(f"  V Mean Avg: {np.mean(data['v_mean']) if data['v_mean'] else 0:.2f}")
            ic(f"  Accuracy HSV: {sum(data['gt_matches_hsv']) / total if total > 0 else 0:.2f} ({sum(data['gt_matches_hsv'])}/{total})")
            ic(f"  Accuracy SVM: {sum(data['gt_matches_svm']) / total if total > 0 else 0:.2f} ({sum(data['gt_matches_svm'])}/{total})")
            ic(f"  Accuracy Final: {sum(data['gt_matches_final']) / total if total > 0 else 0:.2f} ({sum(data['gt_matches_final'])}/{total})")
            ic(f"  Avg Crown Count: {np.mean(data['crown_counts']) if data['crown_counts'] else 0:.2f}")

    def export_results(self, results: List[Dict[str, Any]], filename: str = "results.csv") -> None:
        """
        Export simple image results (filename, score, crowns_found) to a CSV file.

        Args:
            results (List[Dict[str, Any]]): List of result dictionaries.
            filename (str): Filename for the output CSV.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, filename)

            if not results:
                ic("Warning: No results to export.")
                return

            keys = results[0].keys()

            with open(output_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)

            ic(f"Results exported successfully to {output_path}")
        except Exception as e:
            ic(f"Error exporting results: {e}")

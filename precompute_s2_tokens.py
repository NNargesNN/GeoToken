import argparse
import numpy as np
from preprocess import load_annotations  # Assumes it returns a list of tuples: (IMG_ID, LAT, LON, …)
from s2_token_utils import latlng_to_s2_tokens, group_s2_tokens
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Precompute grouped S2 token sequences from a CSV of locations")
    parser.add_argument('--annotations', type=str, default="modules_all/geoclip/model/gps_gallery/coordinates_100K.csv",
                        help="Path to CSV (IMG_ID, LAT, LON, ...)")
    parser.add_argument('--output', type=str, default="data/S2/s2_tokens_100k_grouped_2.npy",
                        help="Output file for grouped S2 tokens")
    parser.add_argument('--level', type=int, default=20, help="S2 level (e.g., 20)")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    grouped_tokens_list = []
    for ann in tqdm(annotations, desc="Computing grouped S2 tokens"):
        try:
            lat = float(ann[1])
            lng = float(ann[2])
        except Exception as e:
            print("Error reading annotation:", ann, e)
            continue
        full_tokens = latlng_to_s2_tokens(lat, lng, args.level)
        grouped = group_s2_tokens(full_tokens, group_size=2)
        grouped_tokens_list.append(grouped)
    grouped_tokens_array = np.array(grouped_tokens_list)  # shape: (N, 1 + (level/4))
    np.save(args.output, grouped_tokens_array)
    print(f"Saved grouped S2 tokens to {args.output}")

if __name__ == "__main__":
    main()

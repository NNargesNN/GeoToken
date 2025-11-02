

import csv

def load_annotations(csv_file):
    """
    Load annotations from a CSV file. If there is an 'IMG_ID' column, return (IMG_ID, LAT, LON).
    Otherwise, return (None, LAT, LON) for each row.
    """
    annotations = []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        # Determine if 'IMG_ID' is in this CSV’s header
        has_id = "IMG_ID" in reader.fieldnames

        for idx, row in enumerate(reader):
            try:
                lat = float(row["LAT"])
                lon = float(row["LON"])
            except KeyError:
                # If LAT/LON columns don't exist, you could raise or skip
                raise KeyError(f"CSV must have 'LAT' and 'LON' columns; missing in row {idx}")
            except ValueError as e:
                # Skip rows with invalid numbers
                print(f"Skipping row {idx} due to invalid LAT/LON: {row} → {e}")
                continue

            if has_id:
                img_id = row["IMG_ID"].replace("/", "_")
            else:
                # Fallback: create a synthetic ID using the row index
                img_id = f"row_{idx}"
            annotations.append((img_id, lat, lon))

    return annotations

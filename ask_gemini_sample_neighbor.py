import os
import re
import ast
import torch
import base64
import argparse
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandarallel import pandarallel
from metrics import compute_metrics  # assumes metrics.py is in PYTHONPATH


def encode_image(image_path: str) -> str:
    """Read an image from disk and return its base64‐encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_coords_from_response(resp_str: str) -> tuple[float, float]:
    """
    Extract (lat, lon) from a JSON‐like string returned by the model.
    Returns (nan, nan) on failure.
    """
    # Regex to find floats
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", resp_str)
    if len(matches) >= 2:
        try:
            return float(matches[0]), float(matches[1])
        except:
            return np.nan, np.nan
    return np.nan, np.nan

def get_response_full(
    image_path: str,
    base_url: str,
    api_key: str,
    model_name: str,
    candidates: list[str],
    samples: list[tuple[float, float]],
    greedy_sample: tuple[float, float],
    detail: str = "low",
    max_tokens: int = 200,
    temperature: float = 0.7,
    n: int = 1,
) -> str:
    """
    Query Gemini with the custom prompt that includes both RAG candidates and LLM samples.
    Returns a single JSON‐string response (we request n=1).
    """
    b64 = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Format candidate list and sample list
    cand_str = "[" + ", ".join(candidates) + "]"
    sample_str = "[" + ", ".join(f"[{lat}, {lon}]" for lat, lon in samples) + "]"
    greedy_sample_str = f"[{greedy_sample[0]}, {greedy_sample[1]}]"
    # ---------------------------
    # Use the provided prompt template:
    # ---------------------------
    prompt_text = f"""
Act as an expert geo-localization analyst. Your primary task is to analyze the provided image and determine its most accurate geographical coordinates.

You are given the following as a model's best guess for this image's location: {greedy_sample_str}. and also these are other guesses for this image's location: {sample_str}.

You **must** provide a location. Do not state that you cannot determine a location.
Your response must be a JSON object in the following format, with no other text:
{{"latitude": float, "longitude": float}}
""".strip() 

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": detail
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n
    }

    resp = requests.post(base_url, headers=headers, json=payload, timeout=(30, 60))
    try:
        choice = resp.json()["choices"][0]
        return choice["message"]["content"]
    except:
        return '{"latitude": 0.0, "longitude": 0.0}'


def process_row_full(
    row: pd.Series,
    base_url: str,
    api_key: str,
    model_name: str,
    root_path: str,
    image_subdir: str,
    n_candidates: int,
    n_samples: int
) -> pd.Series:
    """
    For a given row with IMG_ID, candidate_i_gps, sample_j_lat/lon,
    query Gemini and store 'full_response'.
    """
    img_id = row["IMG_ID"]
    img_path = os.path.join(root_path, image_subdir, img_id)

    # 1) Parse out RAG candidates as "<lat>, <lon>"
    candidates = []
    for i in range(n_candidates):
        raw = row.get(f"candidate_{i}_gps", "")
        try:
            lat, lon = ast.literal_eval(raw)
            candidates.append(f"{lat}, {lon}")
        except:
            candidates.append("0.0, 0.0")

    # 2) Parse out LLM samples as (lat, lon) tuples
    samples = []
    for j in range(n_samples):
        lat = row.get(f"sample_{j}_lat", np.nan)
        lon = row.get(f"sample_{j}_lon", np.nan)
        if not np.isnan(lat) and not np.isnan(lon):
            samples.append((lat, lon))
    greedy_sample = row.get("greedy_lat", np.nan), row.get("greedy_long", np.nan)
    # 3) Call the API
    try:
        resp_str = get_response_full(
            img_path, base_url, api_key, model_name,
            candidates, samples, greedy_sample,
            detail="low", max_tokens=200, temperature=0.7, n=1
        )
    except Exception as e:
        print(f"Error querying {img_id}: {e}")
        resp_str = '{"latitude": 0.0, "longitude": 0.0}'

    row["full_response"] = resp_str
    return row


def compute_and_save_metrics_torch(
    merged_df: pd.DataFrame,
    base_csv: str,
    metrics_output_csv: str,
    root_path: str
):
    """
    - merged_df: DataFrame with columns ["IMG_ID", "full_response"]
    - base_csv: contains ground‐truth "IMG_ID","LAT","LON"
    - metrics_output_csv: where to write aggregated metrics
    We will also save per‐image predictions vs. GT to a CSV.
    """
    # Load GT
    gt_df = pd.read_csv(os.path.join(root_path, base_csv))[["IMG_ID", "LAT", "LON"]]
    gt_df = gt_df.drop_duplicates(subset="IMG_ID").set_index("IMG_ID")

    preds = []
    gts = []
    records = []
    for _, row in merged_df.iterrows():
        img_id = row["IMG_ID"]
        resp_str = row.get("full_response", "")
        pred_lat, pred_lon = extract_coords_from_response(resp_str)
        if img_id in gt_df.index:
            gt_lat, gt_lon = gt_df.loc[img_id, ["LAT", "LON"]]
        else:
            gt_lat, gt_lon = np.nan, np.nan

        # accumulate
        if not np.isnan(pred_lat) and not np.isnan(pred_lon) and not np.isnan(gt_lat):
            preds.append([pred_lat, pred_lon])
            gts.append([gt_lat, gt_lon])

        records.append({
            "IMG_ID": img_id,
            "GT_lat": gt_lat,
            "GT_lon": gt_lon,
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            
        })

    # Save per‐image CSV
    perimg_df = pd.DataFrame(records)
    perimg_csv = metrics_output_csv.replace(".csv", "_per_image.csv")
    perimg_df.to_csv(os.path.join(root_path, perimg_csv), index=False)
    print(f"Per‐image predictions saved to {perimg_csv}")

    # Compute aggregate metrics if we have any valid pairs
    if len(preds) > 0:
        pred_tensor = torch.tensor(preds, dtype=torch.float32)  # shape (N,2)
        gt_tensor = torch.tensor(gts, dtype=torch.float32)
        metrics_dict = compute_metrics(pred_tensor, gt_tensor)
        # Save metrics summary to a small CSV
        summary_df = pd.DataFrame([metrics_dict])
        summary_df.to_csv(os.path.join(root_path, metrics_output_csv), index=False)
        print(f"Metrics summary saved to {metrics_output_csv}")
        print("Metrics:", metrics_dict)
    else:
        print("No valid predictions/ground‐truth pairs to compute metrics.")


def run_for_dataset(
    rag_csv: str,
    base_csv: str,
    samples_csv: str,
    combined_csv: str,
    combined_csv_greedy: str,
    image_subdir: str,
    metrics_output_csv: str,
    n_candidates: int,
    n_samples: int,
    api_key: str,
    model_name: str,
    base_url: str,
    root_path: str,
    nb_workers: int
):
    """
    1) Read combined CSV (which already has columns:
       IMG_ID, candidate_{i}_gps, sample_{j}_lat, sample_{j}_lon).
    2) For every IMG_ID, call process_row_full to get full_response.
    3) Save updated combined CSV.
    4) Compute metrics and save.
    """
    df = pd.read_csv(os.path.join(root_path, combined_csv))
    df_greedy = pd.read_csv(os.path.join(root_path, combined_csv_greedy))

    df_greedy = df_greedy.rename(columns={"sample_0_lat": "greedy_lat", "sample_0_lon": "greedy_long"})
    df_out = df.merge(
    df_greedy[["img_idx", "greedy_lat", "greedy_long"]],
    on="img_idx",
    how="left"
)
    if "full_response" not in df.columns:
        df["full_response"] = np.nan

    # Determine rows to query
    to_query = df[df["full_response"].isna()].copy()
    print(f"{combined_csv}: rows to query = {to_query.shape[0]}")

    # Initialize pandarallel
    pandarallel.initialize(progress_bar=True, nb_workers=nb_workers)

    # Apply in parallel
    updated = to_query.parallel_apply(
        lambda r: process_row_full(
            r, base_url, api_key, model_name,
            root_path, image_subdir,
            n_candidates, n_samples
        ),
        axis=1
    )
    df.update(updated)
    path = os.path.join(root_path, combined_csv).replace(".csv", "_updated.csv")
    df.to_csv(path, index=False)
    print(f"Updated {combined_csv} with full_response.")

    # Compute and save metrics
    compute_and_save_metrics_torch(df, base_csv, metrics_output_csv, root_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # API arguments
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )

    # Paths
    # Paths
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--im2gps_combined_csv_greedy", type=str, default="combined_wo_IM2GPS3K_0.0_1.csv")
    parser.add_argument("--im2gps_combined_csv", type=str, default="combined_wo_epoch_50_final_IM2GPS3K_0.7_30.csv")
    parser.add_argument("--im2gps_base_csv", type=str, default="im2gps3k_places365.csv")
    parser.add_argument("--im2gps_samples_csv", type=str, default="samples_IM2GPS3K_0.7_15.csv")
    parser.add_argument("--im2gps_image_subdir", type=str, default="im2gps3ktest")
    parser.add_argument("--im2gps_metrics_csv", type=str, default="metrics_im2gps3k_choose_gemini_sample_neighbor_0.7_15.csv")

    parser.add_argument("--yfcc_combined_csv_greedy", type=str, default="combined_wo_epoch_50_YFCC4K_0.0_1.csv")
    parser.add_argument("--yfcc_combined_csv", type=str, default="combined_wo_epoch_50_final_YFCC4K_0.7_30.csv")
    parser.add_argument("--yfcc_base_csv", type=str, default="yfcc4k_places.csv")
    parser.add_argument("--yfcc_samples_csv", type=str, default="samples_YFCC4K_0.7_15.csv")
    parser.add_argument("--yfcc_image_subdir", type=str, default="yfcc4k")
    parser.add_argument("--yfcc_metrics_csv", type=str, default="metrics_yfcc4k_choose_gemini_sample_neighbor_0.7_15.csv")

    # RAG + sample counts
    parser.add_argument("--n_candidates", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--nb_workers", type=int, default=20)

    args = parser.parse_args()
    print(args)

    # Run for IM2GPS3K
    run_for_dataset(
        rag_csv=None,  # not needed; combined CSV already has RAG + sample columns
        base_csv=args.im2gps_base_csv,
        samples_csv=None,
        combined_csv=args.im2gps_combined_csv,
        combined_csv_greedy=args.im2gps_combined_csv_greedy,
        image_subdir=args.im2gps_image_subdir,
        metrics_output_csv=args.im2gps_metrics_csv,
        n_candidates=args.n_candidates,
        n_samples=args.n_samples,
        api_key=args.api_key,
        model_name=args.model_name,
        base_url=args.base_url,
        root_path=args.root_path,
        nb_workers=args.nb_workers
    )

    # Run for YFCC4K
    run_for_dataset(
        rag_csv=None,
        base_csv=args.yfcc_base_csv,
        samples_csv=None,
        combined_csv=args.yfcc_combined_csv,
        combined_csv_greedy=args.yfcc_combined_csv_greedy,
        image_subdir=args.yfcc_image_subdir,
        metrics_output_csv=args.yfcc_metrics_csv,
        n_candidates=args.n_candidates,
        n_samples=args.n_samples,
        api_key=args.api_key,
        model_name=args.model_name,
        base_url=args.base_url,
        root_path=args.root_path,
        nb_workers=args.nb_workers
    )

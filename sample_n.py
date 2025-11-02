
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from google import genai

# Dataset‐loading remains unchanged
from train_transformer_c_n_grouped import PrecomputedFeatureTokenDataset

# **Import the updated model**
from model_transformer_add_c_n_grouped import GeoTransformerModelS2

from s2_token_utils import ungroup_s2_tokens, s2_tokens_to_latlng
from metrics import compute_metrics


import os
import pandas as pd
import torch
from geopy.distance import geodesic
from metrics import compute_metrics



def evaluate_closest(samples_csv, ann_csv):
    # samples_csv has columns: img_idx, sample_j, lat, lon, logp, ...
    # ann_csv has columns including LAT, LON
    df_samples = pd.read_csv(samples_csv)
    df_ann     = pd.read_csv(ann_csv)

    # attach the ground‐truth for each sample row
    # it's safe to use iloc because img_idx was the annotation index
    df_samples['true_lat'] = df_samples['img_idx'].map(lambda i: df_ann.iloc[int(i)]['LAT'])
    df_samples['true_lon'] = df_samples['img_idx'].map(lambda i: df_ann.iloc[int(i)]['LON'])


    # just use the first 10 samples per image
    df_samples = df_samples.groupby('img_idx').head(20).reset_index(drop=True)
    # compute geodesic error for every sample
    def row_error(r):
        return geodesic(
            (r['true_lat'], r['true_lon']),
            (r['lat'],      r['lon'])
        ).km
    df_samples['error_km'] = df_samples.apply(row_error, axis=1)

    # for each image, pick the sample with min error
    best = df_samples.loc[df_samples.groupby('img_idx')['error_km'].idxmin()]

    # build tensors for compute_metrics
    preds = torch.tensor(best[['lat','lon']].values, dtype=torch.float)
    trues = torch.tensor(best[['true_lat','true_lon']].values, dtype=torch.float)

    print(f"\n--- Metrics for {os.path.basename(samples_csv)} (closest‐in‐pool) ---")
    print(compute_metrics(preds, trues))


def encode_memory(
    model: GeoTransformerModelS2,
    image_feats: torch.Tensor,        # (B, 768)
    loc_feats:   torch.Tensor,        # (B, 768) 
    text_feats:  torch.Tensor,        # (B, 768) 
    gallery_img_mp16:   torch.Tensor, # (N_mp16, 768) MP16 image‐head
    gallery_loc_mp16:   torch.Tensor, # (N_mp16, 768) MP16 loc‐head
    gallery_txt_mp16:   torch.Tensor, # (N_mp16, 768) MP16 text‐head
    gallery_loc_100k:   torch.Tensor, # (N_100k, 768) 100K loc‐head
    I100k_batch:        torch.LongTensor,  # (B, k100k) indices into 100K gallery
    I16_batch:          torch.LongTensor,  # (B, k16)   indices into MP16 gallery
    metadata_mp16:      np.ndarray,   # (N_mp16, 2) lat/lon (used for forward())
    T100k:              torch.LongTensor,  # (N_100k, seq_len100k)
    T16:                torch.LongTensor   # (N_mp16, seq_len16)
) -> torch.Tensor:
    """
    Build exactly the same “memory” tensor that GeoTransformerModelS2.forward()
    expects. We mimic the retrieval‐token assembly code from model_transformer_add_c_n.py,
    except we do it *ahead of time* for sampling.

    Returns:
        memory: (S, B, d_model)
    """
    B      = image_feats.size(0)
    device = image_feats.device
    d      = model.d_model

    # 1) Project test‐image “heads” into d_model and normalize
    img_emb = model.image_proj(image_feats)  # (B, d_model)
    img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)

    loc_emb = model.loc_proj(loc_feats)      # (B, d_model)
    loc_emb = loc_emb / loc_emb.norm(p=2, dim=-1, keepdim=True)

    txt_emb = model.text_proj(text_feats)    # (B, d_model)
    txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)

    # We will collect two lists:
    #   • “neighbor tokens” from the 100K gallery (prepend loc‐embedding + token embeddings),
    #   • and “neighbor tokens + image+loc+text” from MP16.
    neigh_100k = []
    neigh_mp16 = []

    seq_len100k = T100k.size(1)
    seq_len16   = T16.size(1)

    # # 2) Build “100K gallery” neighbor sequences (prepend loc embedding, then tokens)
    # for j in range(I100k_batch.size(1)):
    #     idx100k = I100k_batch[:, j]            # (B,)
    #     toks100k = T100k[idx100k]              # (B, seq_len100k)

    #     # Get each neighbor’s 100K loc-features from the pre-loaded tensor:
    #     batch_loc100k = gallery_loc_100k[idx100k]  # (B, 768)
    #     emb_loc100k   = model.loc_proj(batch_loc100k)  # (B, d_model)
    #     emb_loc100k   = emb_loc100k / emb_loc100k.norm(p=2, dim=-1, keepdim=True)

    #     # Token‐embedding for all grouped tokens at this position j:
    #     e_list100k = []
    #     for pos in range(seq_len100k):
    #         gid = toks100k[:, pos] + model.token_offsets[pos]   # (B,)
    #         e_list100k.append(model.embedding(gid))             # (B, d_model)
    #     seq_tokens100k = torch.stack(e_list100k, dim=1)        # (B, seq_len100k, d_model)

    #     # Prepend the loc‐embedding as the first “time‐step”:
    #     neighbor_seq100k = torch.cat(
    #         [emb_loc100k.unsqueeze(1), seq_tokens100k],
    #         dim=1
    #     )  # (B, 1 + seq_len100k, d_model)
    #     neigh_100k.append(neighbor_seq100k)

    # 3) Build “MP16 gallery” neighbor sequences (prepend image+loc+text, then tokens)
    for j in range(I16_batch.size(1)):
        idx16 = I16_batch[:, j]          # (B,)
        toks16 = T16[idx16]              # (B, seq_len16)

        nimg = gallery_img_mp16[idx16]   # (B, 768)
        nloc = gallery_loc_mp16[idx16]   # (B, 768)
        ntxt = gallery_txt_mp16[idx16]   # (B, 768)

        emb_nimg = model.image_proj(nimg)   # (B, d_model)
        emb_nimg = emb_nimg / emb_nimg.norm(p=2, dim=-1, keepdim=True)

        emb_nloc = model.loc_proj(nloc)     # (B, d_model)
        emb_nloc = emb_nloc / emb_nloc.norm(p=2, dim=-1, keepdim=True)

        emb_ntxt = model.text_proj(ntxt)    # (B, d_model)
        emb_ntxt = emb_ntxt / emb_ntxt.norm(p=2, dim=-1, keepdim=True)

        e_list16 = []
        for pos in range(seq_len16):
            gid16 = toks16[:, pos] + model.token_offsets[pos]  # (B,)
            e_list16.append(model.retrieval_embedding(gid16))  # (B, d_model)
        seq_tokens16 = torch.stack(e_list16, dim=1)            # (B, seq_len16, d_model)

        # Prepend [emb_nimg, emb_nloc, emb_ntxt]:
        seq_mp16 = torch.cat(
            [emb_nimg.unsqueeze(1),
             emb_nloc.unsqueeze(1),
             emb_ntxt.unsqueeze(1),
             seq_tokens16],
            dim=1
        )  # (B, 3 + seq_len16, d_model)
        neigh_mp16.append(seq_mp16)

    # 4) Flatten “k100k * (1 + seq_len100k)” and “k16 * (3 + seq_len16)”
    # flat100k = torch.stack(neigh_100k, dim=1).view(B, -1, d)   # (B, k100k*(1+seq_len100k), d_model)
    flat16   = torch.stack(neigh_mp16, dim=1).view(B, -1, d)   # (B, k16*(3+seq_len16), d_model)

    # 5) Prepend [CLS, image, loc, text]
    cls_tok   = model.cls_token.expand(1, B, -1)    # (1, B, d_model)
    img_tok   = img_emb.unsqueeze(0)                # (1, B, d_model)
    loc_tok   = loc_emb.unsqueeze(0)                # (1, B, d_model)
    txt_tok   = txt_emb.unsqueeze(0)                # (1, B, d_model)

    # s1 = flat100k.transpose(0, 1)    # (L1, B, d_model)
    s2 = flat16.transpose(0, 1)      # (L2, B, d_model)

    enc_in = torch.cat([cls_tok, img_tok, loc_tok, txt_tok, s2], dim=0)
    enc_in = model.encoder_pos_encoder(enc_in)          # add positional encodings
    return model.encoder(enc_in)                        # (S, B, d_model)


@torch.no_grad()
def sample_pool(
    model: GeoTransformerModelS2,
    memory: torch.Tensor,     # (S, B, d_model)
    num_samples: int,
    temperature: float,
    group_size: int,
    s2_level: int
) -> list:
    """
    Given “memory” = (S, B, d_model), run autoregressive sampling exactly as in your
    model’s decode‐loop. Returns a list‐of‐lists‐of‐dicts:
        [
          [ { 'seq': […], 'lat':…, 'lon':…, 'logp':… }, … ]  # for image 0
          [ { 'seq': […], … }, … ]                            # for image 1
          …
        ]
    """
    S, B, d = memory.shape
    device = memory.device

    # Use exactly the same “start token” shape as the new model:
    start = model.start_token_embedding.unsqueeze(1).expand(1, B, -1)  # (1, B, d_model)
    all_batches = [ [] for _ in range(B) ]

    for _ in range(num_samples):
        decs    = [start]                # list of (t, B, d_model)
        seqs    = [[] for _ in range(B)]
        logps   = [0.0] * B

        for t in range(model.max_seq_length):
            cur = torch.cat(decs, dim=0)                     # (t+1, B, d_model)
            pe  = model.decoder_pos_encoder(cur)             # add positional encoding
            mask= nn.Transformer.generate_square_subsequent_mask(pe.size(0)).to(device)
            out = model.decoder(pe, memory, tgt_mask=mask)   # (t+1, B, d_model)
            last = out[-1]
            if temperature != 0:                                # (B, d_model)
                logits = model.output_layers[t](last) / temperature   # (B, V_t)
                logp   = F.log_softmax(logits, dim=-1)           # (B, V_t)
                probs  = logp.exp()                              # (B, V_t)
                toks   = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
            else:
                logits = model.output_layers[t](last)
                logp   = F.log_softmax(logits, dim=-1)
                toks   = torch.argmax(logp, dim=-1)

            for i in range(B):
                logps[i] += logp[i, toks[i]].item()
                seqs[i].append(int(toks[i].item()))

            gids = toks + model.token_offsets[t]
            emb  = model.embedding(gids).unsqueeze(0)         # (1, B, d_model)
            decs.append(emb)

        # decode lat/lon for each B
        for i in range(B):
            full = ungroup_s2_tokens(seqs[i], group_size)
            lat, lon = s2_tokens_to_latlng(full, s2_level)
            all_batches[i].append({
                'seq': seqs[i],
                'lat': lat,
                'lon': lon,
                'logp': logps[i]
            })

    return all_batches  # list (length B) of lists (length num_samples)


class Location(BaseModel):
    latitude: float
    longitude: float


def judge_single_image(
    client,
    model_name: str,
    prompt: str,
    img_path: str,
    max_retries: int = 3,
    backoff_sec: float = 1.0
) -> tuple:
    """
    Calls Gemini 2.0 Flash Preview, expects JSON {"latitude":<float>,"longitude":<float>}.
    Retries up to max_retries on failure.
    """
    img = Image.open(img_path).convert("RGB")
    for attempt in range(1, max_retries+1):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[prompt, img],
                config={
                  "response_mime_type": "application/json",
                  "response_schema": Location
                }
            )
            if hasattr(resp, "parsed") and resp.parsed:
                return resp.parsed.latitude, resp.parsed.longitude

            txt = resp.text.strip()
            data = json.loads(txt)
            return float(data["latitude"]), float(data["longitude"])
        except Exception:
            if attempt < max_retries:
                time.sleep(backoff_sec * (2**(attempt-1)))
            else:
                raise
    raise RuntimeError("judge_single_image: exhausted retries")


def sample_phase(
    args,
    name: str,
    ann_path: str,
    img_feat_path: str,
    loc_feat_path: str,
    txt_feat_path: str,
    tok_path: str,
    I100k_path: str,
    I16_path: str,
    geo: GeoTransformerModelS2,
    gallery_img_mp16: torch.Tensor,
    gallery_loc_mp16: torch.Tensor,
    gallery_txt_mp16: torch.Tensor,
    gallery_loc_100k: torch.Tensor,
    metadata_mp16: np.ndarray,
    T100k: torch.LongTensor,
    T16:   torch.LongTensor
):
    """
    For a test dataset (YFCC4K or IM2GPS3K):
      1) Load all three test heads (image, loc, text) + tokens + annotations.
      2) For each batch: build memory via encode_memory, sample num_samples.
      3) Save all sampled {img_idx, sample_j, lat, lon, logp} to CSV.
    """
    ds = PrecomputedFeatureTokenDataset(
        ann_path,
        img_feat_path,   # test set image‐head features
        loc_feat_path,   # test set loc‐head features
        txt_feat_path,   # test set text‐head features
        tok_path         # test set tokens
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    I100k = torch.from_numpy(np.load(I100k_path)).long().to(gallery_img_mp16.device)
    I16   = torch.from_numpy(np.load(I16_path)).long().to(gallery_img_mp16.device)

    records = []
    for feats_img, feats_loc, feats_txt, _, lat_b, lon_b, idx_b in tqdm(dl,
                        desc=f"Sampling {name}", total=len(dl)):
        B = feats_img.size(0)
        feats_img = feats_img.to(gallery_img_mp16.device)
        feats_loc = feats_loc.to(gallery_img_mp16.device)
        feats_txt = feats_txt.to(gallery_img_mp16.device)
        idxs = idx_b.tolist()

        mem = encode_memory(
            geo,
            feats_img,      # (B, 768)
            feats_loc,      # (B, 768)
            feats_txt,      # (B, 768)
            gallery_img_mp16,
            gallery_loc_mp16,
            gallery_txt_mp16,
            gallery_loc_100k,
            I100k[idxs, :5],    # take top‐5 from 100K gallery
            I16[idxs,   :15],   # take top‐10 from MP16 gallery
            metadata_mp16,
            T100k,
            T16
        )  # → (S, B, d_model)

        batch_pools = sample_pool(
            geo,
            mem,
            num_samples=args.num_samples,
            temperature=args.temperature,
            group_size=args.group_size,
            s2_level=args.s2_level
        )
        # Each batch_pools[i] is a list of num_samples dicts
        for bi, pools in enumerate(batch_pools):
            img_idx = idxs[bi]
            for j, cand in enumerate(pools):
                records.append({
                    'dataset': name,
                    'img_idx': img_idx,
                    'sample_j': j,
                    'lat': cand['lat'],
                    'lon': cand['lon'],
                    'logp': cand['logp'],
                })

    df = pd.DataFrame(records)
    out_csv = f"samples_wo_epoch_50__{name}_{args.temperature}_{args.num_samples}.csv"
    df.to_csv(out_csv, index=False)
    print(f"→ {name} samples saved to {out_csv}")


def judge_phase(
    args,
    name: str,
    ann_path: str,
    image_folder: str
):
    """
    Given samples in csv ask Gemini to pick the best coordinate.
    """
    samp_csv = f"samples_wo_epoch_50__{name}_{args.temperature}_{args.num_samples}.csv"
    out_csv  = f"results_{name}.csv"
    df_samp  = pd.read_csv(samp_csv)

    df_meta = pd.read_csv(ann_path)
    client  = genai.Client(api_key=args.api_key)

    with open(out_csv, 'w') as f:
        f.write("img_idx,true_lat,true_lon,chosen_lat,chosen_lon\n")

    for img_idx, group in tqdm(df_samp.groupby('img_idx'), desc=f"Judging {name}"):
        row = df_meta.iloc[img_idx]
        true_lat, true_lon = row['LAT'], row['LON']
        img_path = os.path.join(image_folder, row['IMG_ID'])

        candidates = group.sort_values('sample_j')[['lat','lon']].values.tolist()
        txts = "\n".join(f"{j+1}) lat={lat:.5f}, lon={lon:.5f}"
                         for j,(lat,lon) in enumerate(candidates))
        prompt = f"""
You are a world‐class geo‐localization expert.
Below is a photo (path={img_path}) and {len(candidates)} candidate GPS coordinates:
{txts}

Choose the single coordinate from the list that best matches the photo.
If none seem correct, give your own best guess.

**Reply exactly in JSON:**
{{"latitude":<float>,"longitude":<float>}}
""".strip()

        try:
            c_lat, c_lon = judge_single_image(client, args.gemini_model, prompt, img_path)
        except Exception:
            # Fallback to highest log‐prob
            best = group.loc[group['logp'].idxmax()]
            c_lat, c_lon = best['lat'], best['lon']

        with open(out_csv, 'a') as f:
            f.write(f"{img_idx},{true_lat},{true_lon},{c_lat},{c_lon}\n")

    df_r = pd.read_csv(out_csv)
    preds = torch.tensor(df_r[['chosen_lat','chosen_lon']].values)
    trues = torch.tensor(df_r[['true_lat','true_lon']].values)
    print(f"\n=== {name} Gemini‐Judge Metrics ===")
    print(compute_metrics(preds, trues))


def compute_errors_for_subset(df_samples, df_ann, k):
    """
    Given:
      df_samples: DataFrame with ['img_idx','sample_j','lat','lon','logp', ...]
      df_ann:     DataFrame with ['LAT','LON', ...] indexed by img_idx
      k:          take samples with sample_j < k

    Returns:
      best_df: DataFrame of best‐in‐pool sample per img
      metrics_dict: { 'median_error_km', 'mean_error_km', 'rmse_error_km' }
      accuracy_dict: dict returned by compute_metrics(preds, trues)
    """
    df = df_samples.copy()
    df['true_lat'] = df['img_idx'].map(lambda i: df_ann.iloc[int(i)]['LAT'])
    df['true_lon'] = df['img_idx'].map(lambda i: df_ann.iloc[int(i)]['LON'])

    # Keep only first k samples
    df_k = df[df['sample_j'] < k].copy()
    if df_k.empty:
        raise ValueError(f"No samples with sample_j < {k}")

    # Compute geodesic error
    def geodesic_error(row):
        return geodesic(
            (row['true_lat'], row['true_lon']),
            (row['lat'],      row['lon'])
        ).km
    df_k['error_km'] = df_k.apply(geodesic_error, axis=1)

    # Pick best sample for each image
    idx_min = df_k.groupby('img_idx')['error_km'].idxmin()
    best_df = df_k.loc[idx_min].reset_index(drop=True)

    errors = best_df['error_km'].values
    median_error = np.median(errors)
    mean_error   = np.mean(errors)
    rmse_error   = np.sqrt(np.mean(errors ** 2))

    preds = torch.tensor(best_df[['lat','lon']].values, dtype=torch.float)
    trues = torch.tensor(best_df[['true_lat','true_lon']].values, dtype=torch.float)
    accuracy_dict = compute_metrics(preds, trues)

    metrics_dict = {
        'median_error_km': float(median_error),
        'mean_error_km':   float(mean_error),
        'rmse_error_km':   float(rmse_error)
    }

    return best_df, metrics_dict, accuracy_dict


def grid_search_and_evaluate(dataset_name,
                             ann_csv,
                             sample_folder,
                             temperatures=[0.2, 0.5, 0.7, 1.2],
                             k_list=[5, 10, 15, 20, 30]):
    """
    For each temperature T, load samples_wo_epoch_50_<dataset_name>_<T>_30.csv and
    evaluate “closest-in-pool” for k in k_list.
    Save the aggregated results to grid_search_<dataset_name>.csv.
    """
    df_ann = pd.read_csv(ann_csv)

    results = []
    for T in temperatures:
        samples_csv = os.path.join(sample_folder, f"samples_wo_epoch_50__{dataset_name}_{T}_1.csv")
        if not os.path.exists(samples_csv):
            print(f"[Warning] Missing {samples_csv}, skipping T={T}")
            continue

        df_samples = pd.read_csv(samples_csv)

        for k in k_list:
            best_df, metrics_dict, accuracy_dict = compute_errors_for_subset(df_samples, df_ann, k)

            row = {
                'dataset':     dataset_name,
                'temperature': T,
                'k_samples':   k,
                **metrics_dict,
                **accuracy_dict
            }
            results.append(row)

    df_results = pd.DataFrame(results)
    print(f"\n=== Grid Search Results for {dataset_name} ===")
    pd.options.display.float_format = "{:0.3f}".format
    print(df_results)

    out_path = os.path.join(sample_folder, f"grid_search_{dataset_name}.csv")
    df_results.to_csv(out_path, index=False)
    print(f"Saved grid search summary to: {out_path}")

    return df_results

def main():
    parser = argparse.ArgumentParser()
    # GeoTransformer + MP16 gallery + tokens + metadata
    parser.add_argument('--geo_model_path',        type=str,
                        default="modules_all/"
                                "geo_transformer_model_GC_100_MP16_IMf_full_bigger_d_10_text_loss_1e-4_256_grouped_2_without100k_epoch_50.pth")
    parser.add_argument('--features_mp16',         type=str,
                        default="modules_all/"
                                "faiss_outputs/mp16_raw_image_feats.npy")
    parser.add_argument('--loc_features_mp16',     type=str,
                        default="modules_all/"
                                "faiss_outputs/mp16_raw_lochead_feats.npy")
    parser.add_argument('--text_features_mp16',    type=str,
                        default="modules_all/"
                                "faiss_outputs/mp16_raw_text_feats.npy")
    parser.add_argument('--tokens_100k',           type=str,
                        default="data/S2/s2_tokens_100k_grouped_2.npy")
    parser.add_argument('--tokens_mp16',           type=str,
                        default="data/S2/s2_tokens_mp16_grouped_2.npy")
    parser.add_argument('--metadata',              type=str,
                        default="modules_all/"
                                "faiss_outputs/mp16_meta.npy")
    parser.add_argument('--loc_features_100k',     type=str,
                        default="modules_all/"
                                "gallery_retrieval_100k/gps_gallery_loc.npy")

    parser.add_argument('--batch_size',            type=int,   default=512)

    # Gemini
    parser.add_argument('--api_key',               type=str, default=os.environ.get("GOOGLE_API_KEY"))
    parser.add_argument('--gemini_model',          type=str, default='gemini-2.5-flash-preview-05-20')

    # YFCC4K test set
    parser.add_argument('--yfcc_ann',              type=str,
                        default="data/yfcc4k_places.csv")
    parser.add_argument('--yfcc_image_feats',      type=str,
                        default="modules_all/"
                                "faiss_outputs/yfcc_raw_image_feats.npy")
    parser.add_argument('--yfcc_loc_feats',        type=str,
                        default="modules_all/"
                                "faiss_outputs/yfcc_raw_lochead_feats.npy")
    parser.add_argument('--yfcc_text_feats',       type=str,
                        default="modules_all/"
                                "faiss_outputs/yfcc_raw_text_feats.npy")
    parser.add_argument('--yfcc_tokens',           type=str,
                        default="data/S2/s2_tokens_yfcc4k_grouped_2.npy")
    parser.add_argument('--yfcc_folder',           type=str,
                        default="data/yfcc4k")
    parser.add_argument('--yfcc_I100k',            type=str,
                        default="modules_all/"
                                "gallery_retrieval_100k/I_yfcc_100K.npy")
    parser.add_argument('--yfcc_I_mp16',           type=str,
                        default="modules_all/"
                                "faiss_outputs/I_yfcc.npy")

    # IM2GPS3K test set
    parser.add_argument('--im2gps_ann',            type=str,
                        default="data/im2gps3k_places365.csv")
    parser.add_argument('--im2gps_image_feats',    type=str,
                        default="modules_all/"
                                "faiss_outputs/im2gps_raw_image_feats.npy")
    parser.add_argument('--im2gps_loc_feats',      type=str,
                        default="modules_all/"
                                "faiss_outputs/im2gps_raw_lochead_feats.npy")
    parser.add_argument('--im2gps_text_feats',     type=str,
                        default="modules_all/"
                                "faiss_outputs/im2gps_raw_text_feats.npy")
    parser.add_argument('--im2gps_tokens',         type=str,
                        default="data/S2/s2_tokens_im2gps3k_grouped_2.npy")
    parser.add_argument('--im2gps_folder',         type=str,
                        default="data/im2gps3ktest")
    parser.add_argument('--im2gps_I100k',          type=str,
                        default="modules_all/"
                                "gallery_retrieval_100k/I_im2gps_100K.npy")
    parser.add_argument('--im2gps_I_mp16',         type=str,
                        default="modules_all/"
                                "faiss_outputs/I_im2gps.npy")


    # MP16 test set
    parser.add_argument('--mp16_ann',              type=str,
                        default="data/MP16_Pro_filtered.csv")
    parser.add_argument('--mp16_image_feats',      type=str,
                        default="modules_all/"
                                "faiss_outputs/mp16_raw_image_feats.npy")
    parser.add_argument('--mp16_loc_feats',        type=str,
                        default="modules_all/"
                                "faiss_outputs/mp16_raw_lochead_feats.npy")
    parser.add_argument('--mp16_text_feats',       type=str,
                        default="modules_all/"
                                "faiss_outputs/mp16_raw_text_feats.npy")
    parser.add_argument('--mp16_tokens',           type=str,
                        default="data/S2/s2_tokens_mp16_grouped_2.npy")
    parser.add_argument('--mp16_folder',           type=str,
                        default="data/MP16-Pro")
    parser.add_argument('--mp16_I100k',            type=str,
                        default="modules_all/"
                                "gallery_retrieval_100k/I_train_100K.npy")
    parser.add_argument('--mp16_I_mp16',           type=str,
                        default="modules_all/"
                                "faiss_outputs/I_mp16.npy")
    # Sampling settings (we will override in loop)
    parser.add_argument('--num_samples', type=int,   default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--group_size',  type=int,   default=2)
    parser.add_argument('--s2_level',    type=int,   default=20)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load GeoTransformerModelS2
    geo = GeoTransformerModelS2(
        device=device,
        d_model=512,
        transformer_layers=10,
        nhead=8,
        ff_dim=1024,
        dropout=0.1,
        s2_level=args.s2_level,
        group_size=args.group_size,
        max_seq_length=(args.s2_level // args.group_size) + 1
    ).to(device)
    geo.load_state_dict(torch.load(args.geo_model_path, map_location=device))
    geo.eval()

    # Load MP16 gallery features onto GPU + 100K loc features
    gallery_img_mp16 = torch.from_numpy(np.load(args.features_mp16)).float().to(device)
    gallery_loc_mp16 = torch.from_numpy(np.load(args.loc_features_mp16)).float().to(device)
    gallery_txt_mp16 = torch.from_numpy(np.load(args.text_features_mp16)).float().to(device)
    gallery_loc_100k  = torch.from_numpy(np.load(args.loc_features_100k)).float().to(device)

    metadata_mp16 = np.load(args.metadata)  # (N_mp16, 2)
    T100k = torch.from_numpy(np.load(args.tokens_100k)).long().to(device)
    T16   = torch.from_numpy(np.load(args.tokens_mp16)).long().to(device)

    # Define test sets
    test_sets = [
        (
            "YFCC4K",
            args.yfcc_ann,
            args.yfcc_image_feats,
            args.yfcc_loc_feats,
            args.yfcc_text_feats,
            args.yfcc_tokens,
            args.yfcc_folder,
            args.yfcc_I100k,
            args.yfcc_I_mp16
        ),
        (
            "IM2GPS3K",
            args.im2gps_ann,
            args.im2gps_image_feats,
            args.im2gps_loc_feats,
            args.im2gps_text_feats,
            args.im2gps_tokens,
            args.im2gps_folder,
            args.im2gps_I100k,
            args.im2gps_I_mp16
        )
    ]

    # Sampling hyperparameters
    temperatures = [0.0]
    # We will always sample 30 and then later pick first k
    args.num_samples = 1

    # 1) Run sampling for each dataset and each temperature
    for name, ann, img_f, loc_f, txt_f, tok, folder, I100k, I16 in test_sets:
        for T in temperatures:
            print(f"\n********** Sampling for {name}, T={T} **********")
            args.temperature = T
            sample_phase(
                args,
                name,
                ann,
                img_f,
                loc_f,
                txt_f,
                tok,
                I100k,
                I16,
                geo,
                gallery_img_mp16,
                gallery_loc_mp16,
                gallery_txt_mp16,
                gallery_loc_100k,
                metadata_mp16,
                T100k,
                T16
            )

    # 2) After sampling, run grid search and evaluate for each dataset
    sample_folder = os.getcwd()
    for name, ann, *_ in test_sets:
        print(f"\n======== Grid search & evaluation for {name} ========")
        grid_search_and_evaluate(
            dataset_name=name,
            ann_csv=ann,
            sample_folder=sample_folder,
            temperatures=temperatures,
            k_list=[1]
        )

if __name__ == "__main__":
    main()
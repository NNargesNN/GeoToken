#!/usr/bin/env python3
"""
beam_search.py

Use beam search (beam_width=4) over GeoTransformerModelS2 to pick the single best
token sequence (and thus best lat/lon) for every image in YFCC4K and IM2GPS3K.

Usage (defaults already point to your data locations):
    python beam_search.py
"""

import os
import argparse
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# If you want to enable Gemini‐judge, uncomment the next two lines:
# from google import genai

from train_transformer_c_n import PrecomputedFeatureTokenDataset
from model_transformer_add_c_n import GeoTransformerModelS2
from s2_token_utils import ungroup_s2_tokens, s2_tokens_to_latlng
from metrics import compute_metrics

# -----------------------------------------------------------------------------
# 1) Helpers: memory encoding + beam search + (optional) Gemini JSON parsing + retry logic
# -----------------------------------------------------------------------------

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
    Calls Gemini 2.0 Flash Preview, expecting JSON { "latitude": <float>, "longitude": <float> }.
    Retries up to max_retries if errors occur.
    """
    img = Image.open(img_path).convert("RGB")
    for attempt in range(1, max_retries + 1):
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
                time.sleep(backoff_sec * (2 ** (attempt - 1)))
            else:
                raise
    raise RuntimeError("judge_single_image: exhausted retries")


def encode_memory(
    model: GeoTransformerModelS2,
    image_feats:         torch.Tensor,       # (B, 768)
    loc_feats:           torch.Tensor,       # (B, 768)
    text_feats:          torch.Tensor,       # (B, 768)
    gallery_img_mp16:    torch.Tensor,       # (N_mp16, 768)
    gallery_loc_mp16:    torch.Tensor,       # (N_mp16, 768)
    gallery_txt_mp16:    torch.Tensor,       # (N_mp16, 768)
    gallery_loc_100k:    torch.Tensor,       # (N_100k, 768)
    I100k_batch:         torch.LongTensor,   # (B, k100k)
    I16_batch:           torch.LongTensor,   # (B, k16)
    metadata_mp16:       np.ndarray,         # (N_mp16, 2)
    T100k:               torch.LongTensor,   # (N_100k, seq_len100k)
    T16:                 torch.LongTensor    # (N_mp16,  seq_len16)
) -> torch.Tensor:
    """
    Build the encoder "memory" tensor exactly as in model_transformer_add_c.forward().
    Returns a single tensor of shape (S, B, d_model).
    """
    B      = image_feats.size(0)
    device = image_feats.device
    d      = model.d_model

    # 1) Project test‐image heads → d_model, L2‐normalize
    img_emb = model.image_proj(image_feats)  # (B, d_model)
    img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)

    loc_emb = model.loc_proj(loc_feats)      # (B, d_model)
    loc_emb = loc_emb / loc_emb.norm(p=2, dim=-1, keepdim=True)

    txt_emb = model.text_proj(text_feats)    # (B, d_model)
    txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)

    # 2) Build "100K gallery" neighbor sequences
    neigh_100k = []
    seq_len100k = T100k.size(1)
    for j in range(I100k_batch.size(1)):
        idx100k = I100k_batch[:, j]             # (B,)
        toks100k = T100k[idx100k]               # (B, seq_len100k)
        batch_loc100k = gallery_loc_100k[idx100k]  # (B, 768)
        emb_loc100k   = model.loc_proj(batch_loc100k)  # (B, d_model)
        emb_loc100k   = emb_loc100k / emb_loc100k.norm(p=2, dim=-1, keepdim=True)
        e_list100k = []
        for pos in range(seq_len100k):
            gid = toks100k[:, pos] + model.token_offsets[pos]  # (B,)
            e_list100k.append(model.embedding(gid))            # (B, d_model)
        seq100k = torch.stack(e_list100k, dim=1)    
        seq100k = torch.cat([emb_loc100k.unsqueeze(1), seq100k], dim=1) # (B, seq_len100k+1, d_model)
        neigh_100k.append(seq100k)

    # 3) Build "MP16 gallery" neighbor sequences
    neigh_mp16 = []
    seq_len16 = T16.size(1)
    for j in range(I16_batch.size(1)):
        idx16  = I16_batch[:, j]                   # (B,)
        toks16 = T16[idx16]                         # (B, seq_len16)

        nimg = gallery_img_mp16[idx16]             # (B, 768)
        nloc = gallery_loc_mp16[idx16]             # (B, 768)
        ntxt = gallery_txt_mp16[idx16]             # (B, 768)

        emb_nimg = model.image_proj(nimg)          # (B, d_model)
        emb_nimg = emb_nimg / emb_nimg.norm(p=2, dim=-1, keepdim=True)

        emb_nloc = model.loc_proj(nloc)            # (B, d_model)
        emb_nloc = emb_nloc / emb_nloc.norm(p=2, dim=-1, keepdim=True)

        emb_ntxt = model.text_proj(ntxt)           # (B, d_model)
        emb_ntxt = emb_ntxt / emb_ntxt.norm(p=2, dim=-1, keepdim=True)

        e_list16 = []
        for pos in range(seq_len16):
            gid16 = toks16[:, pos] + model.token_offsets[pos]  # (B,)
            e_list16.append(model.retrieval_embedding(gid16))  # (B, d_model)
        seq_tokens16 = torch.stack(e_list16, dim=1)            # (B, seq_len16, d_model)

        # Prepend [emb_nimg, emb_nloc, emb_ntxt]
        seq_mp16 = torch.cat(
            [
              emb_nimg.unsqueeze(1),
              emb_nloc.unsqueeze(1),
              emb_ntxt.unsqueeze(1),
              seq_tokens16
            ],
            dim=1
        )  # (B, 3+seq_len16, d_model)
        neigh_mp16.append(seq_mp16)

    # 4) Flatten "k100k × seq_len100k" and "k16 × (3+seq_len16)"
    # flat100k = torch.stack(neigh_100k, dim=1).view(B, -1, d)  # (B, k100k*seq_len100k, d_model)
    flat16   = torch.stack(neigh_mp16, dim=1).view(B, -1, d)  # (B, k16*(3+seq_len16), d_model)

    # 5) Prepend [CLS; image; loc; text]
    cls_tok = model.cls_token.expand(1, B, -1)     # (1, B, d_model)
    img_tok = img_emb.unsqueeze(0)                 # (1, B, d_model)
    loc_tok = loc_emb.unsqueeze(0)                 # (1, B, d_model)
    txt_tok = txt_emb.unsqueeze(0)                 # (1, B, d_model)

    # s1 = flat100k.transpose(0, 1)                   # (L1, B, d_model)
    s2 = flat16.transpose(0, 1)                     # (L2, B, d_model)

    enc_in = torch.cat([cls_tok, img_tok, loc_tok, txt_tok, s2], dim=0)
    enc_in = model.encoder_pos_encoder(enc_in)      # add positional encodings
    return model.encoder(enc_in)                    # → (S, B, d_model)


def beam_search_batch(
    model: GeoTransformerModelS2,
    memory: torch.Tensor,   # (S, B, d_model)
    beam_width: int,
    group_size: int,
    s2_level: int,
    temperature: float = 1
) -> torch.LongTensor:
    """
    Perform beam search (beam_width=4) *in batch* to get the single best
    grouped‐token sequence of length max_seq_length for each of the B images.

    Returns:
        pred_tokens: LongTensor of shape (B, max_seq_length)
            containing the best token index (0..vocab_size-1) at each position.
    """
    device = memory.device
    B      = memory.size(1)
    d      = model.d_model
    max_seq_length = model.max_seq_length  # total positions (including face)

    # 1) Expand memory so each image's encoding is repeated beam_width times:
    memory_exp = memory.unsqueeze(2)  # (S, B, 1, d)
    memory_exp = memory_exp.expand(-1, -1, beam_width, -1)  # (S, B, beam_width, d)
    memory_exp = memory_exp.contiguous().view(memory_exp.size(0), B * beam_width, d)  # (S, B*beam_width, d)

    # 2) For t=0, we feed only the start token → run through decoder once:
    # The parameter start_token_embedding has shape (1, d_model).
    # Unsqueeze once to (1, 1, d_model), then expand to (1, B, d_model).
    start_emb = model.start_token_embedding  # (1, d_model)
    start_emb = start_emb.unsqueeze(1).expand(-1, B, -1)  # (1, B, d_model)

    with torch.no_grad():
        start_emb_pos = model.decoder_pos_encoder(start_emb) # Add positional encoding
        out0 = model.decoder(start_emb_pos, memory, tgt_mask=None)  # (1, B, d_model)
        last0 = out0[-1]  # (B, d_model)
        logits0 = model.output_layers[0](last0)  # (B, vocab_size0)
        logits0 = logits0 / temperature  # Apply temperature scaling
        logp0 = F.log_softmax(logits0, dim=-1)   # (B, vocab_size0)

    # 3) From logp0, pick top beam_width tokens for each image:
    topk_scores0, topk_ids0 = logp0.topk(beam_width, dim=-1)  # (B, beam_width)

    # 4) Initialize "sequences" and "scores":
    sequences = topk_ids0.unsqueeze(-1).clone()  # (B, beam_width, 1)
    scores    = topk_scores0.clone()             # (B, beam_width)

    # 5) Now run t = 1 … (max_seq_length-1):
    for t in range(1, max_seq_length):
        # Current partial sequences (B * beam_width, t)
        seq_t = sequences.view(B * beam_width, t)  # (B*beam_width, t)

        # Build embeddings for these t tokens
        emb_list = []
        for pos in range(t):
            token_ids = seq_t[:, pos] + model.token_offsets[pos]  # (B*beam_width,)
            emb_pos = model.embedding(token_ids)                   # (B*beam_width, d_model)
            emb_list.append(emb_pos)
        emb_stack = torch.stack(emb_list, dim=0)  # (t, B*beam_width, d_model)

        # Prepare "start token" for every beam:
        #   model.start_token_embedding is (1, d_model).
        #   squeeze → (d_model,), then unsqueeze twice → (1, 1, d_model), then expand to (1, B*beam_width, d_model).
        start_vec = model.start_token_embedding.squeeze(0)             # (d_model,)
        start_rep = start_vec.unsqueeze(0).unsqueeze(1).expand(1, B * beam_width, d)  # (1, B*beam_width, d)

        decoder_input = torch.cat([start_rep, emb_stack], dim=0)       # (t+1, B*beam_width, d_model)
        decoder_input = model.decoder_pos_encoder(decoder_input) # Add positional encoding
        tgt_mask     = nn.Transformer.generate_square_subsequent_mask(t + 1).to(device)

        with torch.no_grad():
            dec_out  = model.decoder(decoder_input, memory_exp, tgt_mask=tgt_mask)  # (t+1, B*beam_width, d_model)
            last     = dec_out[-1]                                                   # (B*beam_width, d_model)
            logits_t = model.output_layers[t](last)                                  # (B*beam_width, vocab_size_t)
            logits_t = logits_t / temperature  # Apply temperature scaling
            logp_t   = F.log_softmax(logits_t, dim=-1)                               # (B*beam_width, vocab_size_t)

        vocab_size_t     = model.target_token_vocab_sizes[t]
        logp_t_reshaped  = logp_t.view(B, beam_width, vocab_size_t)  # (B, beam_width, vocab_size_t)
        prev_scores      = scores.unsqueeze(-1)                      # (B, beam_width, 1)
        total_scores     = prev_scores + logp_t_reshaped             # (B, beam_width, vocab_size_t)

        flat_scores = total_scores.view(B, beam_width * vocab_size_t)    # (B, beam_width * vocab_size_t)
        topk_val, topk_idx = flat_scores.topk(beam_width, dim=-1)         # (B, beam_width)

        prev_beam_idx  = topk_idx // vocab_size_t  # (B, beam_width)
        new_token_idx  = topk_idx % vocab_size_t   # (B, beam_width)

        old_seq             = sequences  # (B, beam_width, t)
        prev_beam_idx_exp   = prev_beam_idx.unsqueeze(-1).expand(-1, -1, t)  # (B, beam_width, t)
        gathered            = old_seq.gather(1, prev_beam_idx_exp)          # (B, beam_width, t)
        new_token_idx_exp   = new_token_idx.unsqueeze(-1)                   # (B, beam_width, 1)

        sequences = torch.cat([gathered, new_token_idx_exp], dim=-1)        # (B, beam_width, t+1)
        scores    = topk_val                                               # (B, beam_width)

    best_beam_indices = scores.argmax(dim=-1)  # (B,)
    best_beams        = sequences[torch.arange(B), best_beam_indices, :]  # (B, max_seq_length)

    return best_beams  # (B, max_seq_length)


# -----------------------------------------------------------------------------
# 2) Two‐phase main: beam search → (optional) judge
# -----------------------------------------------------------------------------

def beam_phase(
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
    For test set 'name', run a full‐batch beam search (beam_width=4) for each image.
    Writes out a CSV "best_<name>.csv" with columns:
      img_idx, true_lat, true_lon, pred_lat, pred_lon
    """
    ds = PrecomputedFeatureTokenDataset(
        ann_path,
        img_feat_path,   # test‐set image‐head features
        loc_feat_path,   # test‐set loc‐head features
        txt_feat_path,   # test‐set text‐head features
        tok_path         # test‐set tokens
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    I100k = torch.from_numpy(np.load(I100k_path)).long().to(gallery_img_mp16.device)
    I16   = torch.from_numpy(np.load(I16_path)).long().to(gallery_img_mp16.device)

    records = []
    for feats_img, feats_loc, feats_txt, _, lat_b, lon_b, idx_b in tqdm(dl,
                        desc=f"Beam‐search {name}", total=len(dl)):
        B = feats_img.size(0)
        feats_img = feats_img.to(gallery_img_mp16.device)
        feats_loc = feats_loc.to(gallery_img_mp16.device)
        feats_txt = feats_txt.to(gallery_img_mp16.device)
        idxs      = idx_b.tolist()

        # 1) Build memory for this batch:
        mem = encode_memory(
            geo,
            feats_img,            # (B, 768)
            feats_loc,            # (B, 768)
            feats_txt,            # (B, 768)
            gallery_img_mp16,     # (N_mp16, 768)
            gallery_loc_mp16,     # (N_mp16, 768)
            gallery_txt_mp16,     # (N_mp16, 768)
            gallery_loc_100k,     # (N_100k, 768)
            I100k[idxs, :5],      # (B, 5)
            I16[idxs,   :15],      # (B, 5)
            metadata_mp16,        # (N_mp16, 2)
            T100k,                # (N_100k, seq_len100k)
            T16                   # (N_mp16,   seq_len16)
        )  # → (S, B, d_model)

        # 2) Run batch beam search:
        best_beams = beam_search_batch(
            geo,
            mem,
            beam_width=5,
            group_size=args.group_size,
            s2_level=args.s2_level,
            temperature=args.temperature
        )  # (B, max_seq_length)

        # 3) Convert each beam (grouped tokens) → lat/lon
        for i in range(B):
            img_idx  = idxs[i]
            true_lat = float(lat_b[i].item())
            true_lon = float(lon_b[i].item())

            token_list = best_beams[i].tolist()
            full_tokens = ungroup_s2_tokens(token_list, group_size=args.group_size)
            pred_lat, pred_lon = s2_tokens_to_latlng(full_tokens, args.s2_level)

            records.append({
                'dataset': name,
                'img_idx': img_idx,
                'true_lat': true_lat,
                'true_lon': true_lon,
                'pred_lat': pred_lat,
                'pred_lon': pred_lon
            })

    df = pd.DataFrame(records)
    out_csv = f"best_{name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"→ {name} beam‐search results saved to {out_csv}")

    # 4) Compute metrics:
    preds = torch.tensor(df[['pred_lat','pred_lon']].values)
    trues = torch.tensor(df[['true_lat','true_lon']].values)
    print(f"\n=== {name} Beam‐Search Metrics ===")
    print(compute_metrics(preds, trues))


# -----------------------------------------------------------------------------
# 3) main()
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # GeoTransformer + MP16 gallery + tokens + metadata
    parser.add_argument('--geo_model_path',        type=str,
                        default="modules_all/"
                                "geo_transformer_model_GC_100_MP16_IMf_full_bigger_d_10_text_loss_1e-4_256_grouped_2_without100k_epoch_50.pth")
    parser.add_argument('--features_mp16',     type=str,
                        default='modules_all/'
                                'faiss_outputs/mp16_raw_image_feats.npy')
    parser.add_argument('--loc_features_mp16', type=str,
                        default='modules_all/'
                                'faiss_outputs/mp16_raw_lochead_feats.npy')
    parser.add_argument('--text_features_mp16',type=str,
                        default='modules_all/'
                                'faiss_outputs/mp16_raw_text_feats.npy')
    parser.add_argument('--tokens_100k',       type=str,
                        default='data/S2/s2_tokens_100k_grouped_2.npy')
    parser.add_argument('--tokens_mp16',       type=str,
                        default='data/S2/s2_tokens_mp16_grouped_2.npy')
    parser.add_argument('--metadata',          type=str,
                        default='modules_all/'
                                'faiss_outputs/mp16_meta.npy')
    parser.add_argument('--gallery_loc_100k',  type=str,
                        default='modules_all/'
                                'gallery_retrieval_100k/gps_gallery_loc.npy')
    parser.add_argument('--batch_size',        type=int, default=512)
    parser.add_argument('--group_size',        type=int, default=2)
    parser.add_argument('--s2_level',          type=int, default=20)
    parser.add_argument('--temperature',       type=float, default=1, help='Temperature for beam search softmax')

    # YFCC4K test set
    parser.add_argument('--yfcc_ann',          type=str,
                        default='data/yfcc4k_places.csv')
    parser.add_argument('--yfcc_image_feats',  type=str,
                        default='modules_all/'
                                'faiss_outputs/yfcc_raw_image_feats.npy')
    parser.add_argument('--yfcc_loc_feats',    type=str,
                        default='modules_all/'
                                'faiss_outputs/yfcc_raw_lochead_feats.npy')
    parser.add_argument('--yfcc_text_feats',   type=str,
                        default='modules_all/'
                                'faiss_outputs/yfcc_raw_text_feats.npy')
    parser.add_argument('--yfcc_tokens',       type=str,
                        default='data/S2/s2_tokens_yfcc4k_grouped_2.npy')
    parser.add_argument('--yfcc_folder',       type=str,
                        default='data/yfcc4k')
    parser.add_argument('--yfcc_I100k',        type=str,
                        default='modules_all/'
                                'gallery_retrieval_100k/I_yfcc_100K.npy')
    parser.add_argument('--yfcc_I_mp16',       type=str,
                        default='modules_all/'
                                'faiss_outputs/I_yfcc.npy')

    # IM2GPS3K test set
    parser.add_argument('--im2gps_ann',        type=str,
                        default='data/im2gps3k_places365.csv')
    parser.add_argument('--im2gps_image_feats',type=str,
                        default='modules_all/'
                                'faiss_outputs/im2gps_raw_image_feats.npy')
    parser.add_argument('--im2gps_loc_feats',  type=str,
                        default='modules_all/'
                                'faiss_outputs/im2gps_raw_lochead_feats.npy')
    parser.add_argument('--im2gps_text_feats', type=str,
                        default='modules_all/'
                                'faiss_outputs/im2gps_raw_text_feats.npy')
    parser.add_argument('--im2gps_tokens',     type=str,
                        default='data/S2/s2_tokens_im2gps3k_grouped_2.npy')
    parser.add_argument('--im2gps_folder',     type=str,
                        default='data/im2gps3ktest')
    parser.add_argument('--im2gps_I100k',      type=str,
                        default='modules_all/'
                                'gallery_retrieval_100k/I_im2gps_100K.npy')
    parser.add_argument('--im2gps_I_mp16',     type=str,
                        default='modules_all/'
                                'faiss_outputs/I_im2gps.npy')

    # (Optional) Gemini‐judge flags
    parser.add_argument('--api_key',           type=str,
                        default=os.environ.get("GOOGLE_API_KEY", None))
    parser.add_argument('--gemini_model',      type=str,
                        default='gemini-2.5-flash-preview-05-20')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load GeoTransformerModelS2
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

    # 2) Load MP16 "gallery" heads + metadata + tokens
    gallery_img_mp16 = torch.from_numpy(np.load(args.features_mp16)).float().to(device)
    gallery_loc_mp16 = torch.from_numpy(np.load(args.loc_features_mp16)).float().to(device)
    gallery_txt_mp16 = torch.from_numpy(np.load(args.text_features_mp16)).float().to(device)
    gallery_loc_100k = torch.from_numpy(np.load(args.gallery_loc_100k)).float().to(device)
    metadata_mp16    = np.load(args.metadata)  # (N_mp16, 2)
    T100k            = torch.from_numpy(np.load(args.tokens_100k)).long().to(device)  # (N_100k, seq_len100k)
    T16              = torch.from_numpy(np.load(args.tokens_mp16)).long().to(device)   # (N_mp16,   seq_len16)

    # 3) Run beam search on YFCC4K
    print("\n>>> RUNNING BEAM SEARCH on YFCC4K …")
    beam_phase(
        args,
        name="YFCC4K",
        ann_path=args.yfcc_ann,
        img_feat_path=args.yfcc_image_feats,
        loc_feat_path=args.yfcc_loc_feats,
        txt_feat_path=args.yfcc_text_feats,
        tok_path=args.yfcc_tokens,
        I100k_path=args.yfcc_I100k,
        I16_path=args.yfcc_I_mp16,
        geo=geo,
        gallery_img_mp16=gallery_img_mp16,
        gallery_loc_mp16=gallery_loc_mp16,
        gallery_txt_mp16=gallery_txt_mp16,
        gallery_loc_100k=gallery_loc_100k,
        metadata_mp16=metadata_mp16,
        T100k=T100k,
        T16=T16
    )

    # 4) Run beam search on IM2GPS3K
    print("\n>>> RUNNING BEAM SEARCH on IM2GPS3K …")
    beam_phase(
        args,
        name="IM2GPS3K",
        ann_path=args.im2gps_ann,
        img_feat_path=args.im2gps_image_feats,
        loc_feat_path=args.im2gps_loc_feats,
        txt_feat_path=args.im2gps_text_feats,
        tok_path=args.im2gps_tokens,
        I100k_path=args.im2gps_I100k,
        I16_path=args.im2gps_I_mp16,
        geo=geo,
        gallery_img_mp16=gallery_img_mp16,
        gallery_loc_mp16=gallery_loc_mp16,
        gallery_txt_mp16=gallery_txt_mp16,
        gallery_loc_100k=gallery_loc_100k,
        metadata_mp16=metadata_mp16,
        T100k=T100k,
        T16=T16
    )


if __name__ == '__main__':
    main()

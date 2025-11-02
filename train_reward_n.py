#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
torch_import = True
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_transformer_add_c_n_grouped import GeoTransformerModelS2
from s2_token_utils import latlng_to_s2_tokens, group_s2_tokens
from metrics import compute_metrics



# Distance thresholds (km)
THRESHOLDS = [200]
NUM_BINS = len(THRESHOLDS) + 1
R_EARTH = 6371.0

# ----------------------------------------------------------------------------
# Utility: haversine distance between two (lat, lon) tensors
# ----------------------------------------------------------------------------
def haversine_tensor(lat1, lon1, lat2, lon2):
    phi1 = torch.deg2rad(lat1)
    phi2 = torch.deg2rad(lat2)
    dphi = phi2 - phi1
    dlambda = torch.deg2rad(lon2 - lon1)
    a = torch.sin(dphi/2)**2 + torch.cos(phi1)*torch.cos(phi2)*torch.sin(dlambda/2)**2
    return R_EARTH * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

# ----------------------------------------------------------------------------
# Reward head wrapping the pretrained GeoTransformerModelS2
# ----------------------------------------------------------------------------
class RewardScorer(nn.Module):
    def __init__(self, geo: GeoTransformerModelS2, hidden_dim: int = 128):
        super().__init__()
        self.geo = geo
        # two-layer head predicting bins
        self.fc1 = nn.Linear(geo.d_model, hidden_dim)
        self.head = nn.Linear(hidden_dim, NUM_BINS)

    def encode_memory(
        self,
        img_feats, loc_feats, txt_feats,
        gallery_feats, gallery_loc_feats, gallery_txt_feats,
        neigh_idx, neigh_tokens
    ):
        B = img_feats.size(0)
        # project query
        img_e = self.geo.image_proj(img_feats)
        img_e = img_e / img_e.norm(p=2, dim=-1, keepdim=True)
        loc_e = self.geo.loc_proj(loc_feats)
        loc_e = loc_e / loc_e.norm(p=2, dim=-1, keepdim=True)
        txt_e = self.geo.text_proj(txt_feats)
        txt_e = txt_e / txt_e.norm(p=2, dim=-1, keepdim=True)

        neigh_seqs = []
        for j in range(neigh_idx.size(1)):
            toks = neigh_tokens[:, j, :]         # (B, seq_len)
            nimg = gallery_feats[:, j, :]        # (B, 768)
            nloc = gallery_loc_feats[:, j, :]    # (B, 768)
            ntxt = gallery_txt_feats[:, j, :]    # (B, 768)
            e_img = self.geo.image_proj(nimg)
            e_img = e_img / e_img.norm(p=2, dim=-1, keepdim=True)
            e_loc = self.geo.loc_proj(nloc)
            e_loc = e_loc / e_loc.norm(p=2, dim=-1, keepdim=True)
            e_txt = self.geo.text_proj(ntxt)
            e_txt = e_txt / e_txt.norm(p=2, dim=-1, keepdim=True)
            # token embeddings
            token_embs = []
            for pos in range(toks.size(1)):
                ids = toks[:, pos] + self.geo.token_offsets[pos]
                token_embs.append(self.geo.retrieval_embedding(ids))
            seq = torch.stack(token_embs, dim=1)
            seq = torch.cat([e_img.unsqueeze(1), e_loc.unsqueeze(1), e_txt.unsqueeze(1), seq], dim=1)
            neigh_seqs.append(seq)

        flat = torch.stack(neigh_seqs, dim=1).view(B, -1, self.geo.d_model)
        cls_tok = self.geo.cls_token.expand(B, -1).unsqueeze(0)
        img_tok = img_e.unsqueeze(0)
        loc_tok = loc_e.unsqueeze(0)
        txt_tok = txt_e.unsqueeze(0)
        enc_seq = torch.cat([cls_tok, img_tok, loc_tok, txt_tok, flat.transpose(0,1)], dim=0)
        enc_seq = self.geo.encoder_pos_encoder(enc_seq)
        return self.geo.encoder(enc_seq)  # (S, B, d_model)

    def forward(
        self,
        img_feats, loc_feats, txt_feats,
        gallery_feats, gallery_loc_feats, gallery_txt_feats,
        neigh_idx, neigh_tokens,
        sample_tokens
    ):
        device = img_feats.device
        B = img_feats.size(0)
        # build memory
        mem = self.encode_memory(
            img_feats, loc_feats, txt_feats,
            gallery_feats, gallery_loc_feats, gallery_txt_feats,
            neigh_idx, neigh_tokens
        )
        # decoder teacher forcing
        dec_inputs = [self.geo.start_token_embedding.expand(B, -1)]
        for t in range(sample_tokens.size(1)):
            ids = sample_tokens[:, t] + self.geo.token_offsets[t]
            dec_inputs.append(self.geo.embedding(ids))
        dec_in = torch.stack(dec_inputs, dim=0)
        dec_in = self.geo.decoder_pos_encoder(dec_in)
        mask = nn.Transformer.generate_square_subsequent_mask(dec_in.size(0)).to(device)
        out = self.geo.decoder(dec_in, mem, tgt_mask=mask)
        last = out[-1]  # (B, d_model)
        h = F.relu(self.fc1(last))
        return self.head(h)  # (B, NUM_BINS)

# ----------------------------------------------------------------------------
# Dataset loading sample candidates and pretrained features
# ----------------------------------------------------------------------------
class RewardDataset(Dataset):
    def __init__(
        self,
        samples_csv,
        features_mp16_file,
        loc_feats_mp16_file,
        text_feats_mp16_file,
        tokens_mp16_file,
        neigh_idx_file,
        metadata_file,
        s2_level,
        group_size,
        mode='train',
        num_neighbors=15
    ):
        df = pd.read_csv(samples_csv)
        df = df.iloc[:10000000]
        self.groups = [g.reset_index(drop=True)
                       for _, g in df.groupby('img_idx')]
        self.metadata = np.load(metadata_file)  # (N_meta,2)
        # load gallery
        self.mode = mode
        self.num_neighbors = num_neighbors
        self.feats = torch.from_numpy(np.load(features_mp16_file)).float()
        self.loc_feats = torch.from_numpy(np.load(loc_feats_mp16_file)).float()
        self.txt_feats = torch.from_numpy(np.load(text_feats_mp16_file)).float()
        self.neigh_idx_all = torch.from_numpy(np.load(neigh_idx_file)).long()
        self.neigh_tokens = torch.from_numpy(np.load(tokens_mp16_file)).long()
        self.s2_level = s2_level
        self.group_size = group_size

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        grp = self.groups[i]
        lats = grp['lat'].astype(float).values
        lons = grp['lon'].astype(float).values
        # convert samples to tokens
        seqs = []
        for lat, lon in zip(lats, lons):
            full = latlng_to_s2_tokens(lat, lon, self.s2_level)
            grp_tok = group_s2_tokens(full, self.group_size)
            seqs.append(grp_tok)
        sample_tokens = torch.tensor(seqs, dtype=torch.long)  # (N, S)
        img_idx = grp['img_idx'].iloc[0]
        # gallery neighbor info
        raw_idx = self.neigh_idx_all[img_idx]
        if self.mode == 'train':
            neigh_idx = raw_idx[1:1 + self.num_neighbors]
        else:
            neigh_idx = raw_idx[:self.num_neighbors]
        # true lat/lon
        true = torch.tensor(self.metadata[img_idx], dtype=torch.float)
        return {
            'feat': self.feats[img_idx],
            'loc_feat': self.loc_feats[img_idx],
            'txt_feat': self.txt_feats[img_idx],
            'gallery_feats': self.feats[neigh_idx],
            'gallery_loc_feats': self.loc_feats[neigh_idx],
            'gallery_txt_feats': self.txt_feats[neigh_idx],
            'neigh_idx': neigh_idx,
            'neigh_tokens': self.neigh_tokens[neigh_idx],
            'sample_tokens': sample_tokens,
            'lats': torch.tensor(lats),
            'lons': torch.tensor(lons),
            'true': true
        }

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train_reward(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load pretrained transformer
    geo = GeoTransformerModelS2(
        device='cuda', d_model=args.sim_dim,
        transformer_layers=args.transformer_layers,
        nhead=args.nhead, ff_dim=args.ff_dim,
        dropout=args.dropout, s2_level=args.s2_level,
        group_size=args.group_size,
        max_seq_length=args.max_seq_length,
        num_neighbors=args.num_neighbors
    )
    geo.load_state_dict(torch.load(args.transformer_output, map_location='cpu'))
    geo.to(device).eval()
    # for p in geo.parameters(): p.requires_grad = False
    reward = RewardScorer(geo, hidden_dim=args.hidden_dim).to(device)

    ds = RewardDataset(
        samples_csv=args.train_samples_csv,
        features_mp16_file=args.features_mp16_file,
        loc_feats_mp16_file=args.loc_feats_mp16_file,
        text_feats_mp16_file=args.text_feats_mp16_file,
        tokens_mp16_file=args.tokens_mp16_file,
        neigh_idx_file=args.retrieval_indices_file_mp16,
        metadata_file=args.metadata_file,
        s2_level=args.s2_level,
        group_size=args.group_size,
        mode='train',
        num_neighbors=args.num_neighbors
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(reward.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        total_loss = 0.0
        for batch in loop:
            B = batch['sample_tokens'].size(0)
            N = batch['sample_tokens'].size(1)
            # to device
            img_f = batch['feat'].to(device)
            loc_f = batch['loc_feat'].to(device)
            txt_f = batch['txt_feat'].to(device)
            gallery_f = batch['gallery_feats'].to(device)
            gallery_loc = batch['gallery_loc_feats'].to(device)
            gallery_txt = batch['gallery_txt_feats'].to(device)
            neigh_idx = batch['neigh_idx'].to(device)
            neigh_tok = batch['neigh_tokens'].to(device)
            toks = batch['sample_tokens'].to(device)
            lats = batch['lats'].to(device)
            lons = batch['lons'].to(device)
            true = batch['true'].to(device)
            # flatten for scoring
            feats_flat = img_f.unsqueeze(1).expand(-1,N,-1).reshape(B*N,-1)
            loc_flat = loc_f.unsqueeze(1).expand(-1,N,-1).reshape(B*N,-1)
            txt_flat = txt_f.unsqueeze(1).expand(-1,N,-1).reshape(B*N,-1)
            toks_flat = toks.reshape(B*N, -1)
            idx_flat = neigh_idx.unsqueeze(1).expand(-1,N,-1).reshape(B*N, -1)
            tok_neigh = neigh_tok  # global for all
            # forward
            logits_flat = reward(
                feats_flat, loc_flat, txt_flat,
                gallery_f, gallery_loc, gallery_txt,
                idx_flat, tok_neigh,
                toks_flat
            )  # (B*N, NUM_BINS)
            # compute labels
            d_km = haversine_tensor(
                true[:,0].unsqueeze(1).expand(-1,N),
                true[:,1].unsqueeze(1).expand(-1,N),
                lats, lons
            )
            th = torch.tensor(THRESHOLDS, device=d_km.device)
            labels = torch.bucketize(d_km, th, right=True)
            loss = F.cross_entropy(logits_flat, labels.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch {epoch+1} avg loss = {total_loss/len(loader):.4f}")
        sch.step()
    torch.save(reward.state_dict(), args.save_path)
    print("Training complete.")

# ----------------------------------------------------------------------------
# Evaluation on a sample set
# ----------------------------------------------------------------------------
def evaluate_set(args, samples_csv, name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load models
    geo = GeoTransformerModelS2(
        device='cuda', d_model=args.sim_dim,
        transformer_layers=args.transformer_layers,
        nhead=args.nhead, ff_dim=args.ff_dim,
        dropout=args.dropout, s2_level=args.s2_level,
        group_size=args.group_size,
        max_seq_length=args.max_seq_length,
        num_neighbors=args.num_neighbors
    )
    geo.load_state_dict(torch.load(args.transformer_output, map_location='cpu'))
    geo.to(device).eval()
    reward = RewardScorer(geo, hidden_dim=args.hidden_dim).to(device)
    reward.load_state_dict(torch.load(args.save_path, map_location=device))
    reward.eval()
    ds = RewardDataset(
        samples_csv=samples_csv,
        features_mp16_file=args.features_mp16_file,
        loc_feats_mp16_file=args.loc_feats_mp16_file,
        text_feats_mp16_file=args.text_feats_mp16_file,
        tokens_mp16_file=args.tokens_mp16_file,
        neigh_idx_file=args.retrieval_indices_file_mp16,
        metadata_file=args.metadata_file,
        s2_level=args.s2_level,
        group_size=args.group_size,
        mode='eval',
        num_neighbors=args.num_neighbors
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    chosen, true_coords = [], []
    for batch in tqdm(loader, desc=f"Eval {name}"):
        B = batch['sample_tokens'].size(0)
        N = batch['sample_tokens'].size(1)
        # to device
        img_f = batch['feat'].to(device)
        loc_f = batch['loc_feat'].to(device)
        txt_f = batch['txt_feat'].to(device)
        gallery_f = batch['gallery_feats'].to(device)
        gallery_loc = batch['gallery_loc_feats'].to(device)
        gallery_txt = batch['gallery_txt_feats'].to(device)
        neigh_idx = batch['neigh_idx'].to(device)
        neigh_tok = batch['neigh_tokens'].to(device)
        toks = batch['sample_tokens'].to(device)
        lats = batch['lats'].to(device)
        lons = batch['lons'].to(device)
        true = batch['true']
        # flatten
        feats_flat = img_f.unsqueeze(1).expand(-1,N,-1).reshape(B*N,-1)
        loc_flat = loc_f.unsqueeze(1).expand(-1,N,-1).reshape(B*N,-1)
        txt_flat = txt_f.unsqueeze(1).expand(-1,N,-1).reshape(B*N,-1)
        toks_flat = toks.reshape(B*N,-1)
        idx_flat = neigh_idx.unsqueeze(1).expand(-1,N,-1).reshape(B*N,-1)
        with torch.no_grad():
            logits_flat = reward(
                feats_flat, loc_flat, txt_flat,
                gallery_f, gallery_loc, gallery_txt,
                idx_flat, neigh_tok,
                toks_flat
            )
        logits = logits_flat.view(B, N, NUM_BINS)
        probs = F.softmax(logits, dim=2)
        pred_bins = probs.argmax(dim=2)
        min_bin = pred_bins.min(dim=1).values
        for i in range(B):
            b = min_bin[i]
            mask = pred_bins[i] == b
            confidences = torch.where(mask, probs[i,:,b], torch.tensor(-1., device=device))
            j = confidences.argmax().item()
            chosen.append((lats[i,j].item(), lons[i,j].item()))
            true_coords.append((true[i,0].item(), true[i,1].item()))
    preds = torch.tensor(chosen)
    trues = torch.tensor(true_coords)
    print(f"\n=== Ranking Metrics ({name}) ===")
    print(compute_metrics(preds, trues))

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train','eval'], default='train')
    p.add_argument('--train_samples_csv', default='samples_wo_epoch_50_MP16_0.7_10.csv')
    p.add_argument('--im2gps_samples_csv', default='samples_wo_epoch_50_IM2GPS3K_0.7_10.csv')
    p.add_argument('--yfcc_samples_csv', default='samples_wo_epoch_50_YFCC4K_0.7_10.csv')
    p.add_argument('--metadata_file', default='faiss_outputs/mp16_meta.npy')
    p.add_argument('--features_mp16_file', default='faiss_outputs/mp16_raw_image_feats.npy')
    p.add_argument('--loc_feats_mp16_file', default='faiss_outputs/mp16_raw_lochead_feats.npy')
    p.add_argument('--text_feats_mp16_file', default='faiss_outputs/mp16_raw_text_feats.npy')
    p.add_argument('--retrieval_indices_file_mp16', default='faiss_outputs/I_mp16.npy')
    p.add_argument('--tokens_mp16_file', default='data/S2/s2_tokens_mp16_grouped_2.npy')
    p.add_argument('--transformer_output',         type=str, default="geo_transformer_model_GC_100_MP16_IMf_full_bigger_d_10_text_loss_1e-4_256_grouped_2_without100k_epoch_50.pth")
    p.add_argument('--save_path', default='reward_scorer_v2_full_now.pth')
    p.add_argument('--hidden_dim', type=int, default=2048)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--step_size', type=int, default=1)
    p.add_argument('--gamma', type=float, default=0.9)
    p.add_argument('--s2_level', type=int, default=20)
    p.add_argument('--group_size', type=int, default=2)
    p.add_argument('--sim_dim', type=int, default=512)
    p.add_argument('--transformer_layers', type=int, default=10)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--ff_dim', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--max_seq_length', type=int, default=11)
    p.add_argument('--num_neighbors', type=int, default=15)
    args = p.parse_args()

    if args.mode == 'train':
        train_reward(args)
    else:
        evaluate_set(args, args.im2gps_samples_csv, 'im2gps3k')
        evaluate_set(args, args.yfcc_samples_csv, 'yfcc4k')

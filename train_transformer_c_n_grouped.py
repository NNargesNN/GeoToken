import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast

from preprocess import load_annotations
from model_transformer_add_c_n_grouped import GeoTransformerModelS2
from metrics import compute_metrics  
from s2_token_utils import ungroup_s2_tokens, s2_tokens_to_latlng

# Dataset for training/evaluation.
class PrecomputedFeatureTokenDataset(Dataset):
    def __init__(self, annotations_file, features_file, loc_features_file, text_features_file, tokens_file):
        self.annotations = load_annotations(annotations_file)
        self.features = np.load(features_file)  # (N, 768)
        self.loc_features = np.load(loc_features_file)  # (N, 768)
        self.text_features = np.load(text_features_file)  # (N, 768)
        self.tokens = np.load(tokens_file)      # (N, seq_len) where seq_len = grouped token sequence length
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        feature = torch.tensor(self.features[idx], dtype=torch.float)
        loc_feature = torch.tensor(self.loc_features[idx], dtype=torch.float)
        text_feature = torch.tensor(self.text_features[idx], dtype=torch.float)
        tokens = torch.tensor(self.tokens[idx], dtype=torch.long)
        lat = float(ann[1])
        lon = float(ann[2])
        return feature, loc_feature, text_feature, tokens, torch.tensor(lat, dtype=torch.float), torch.tensor(lon, dtype=torch.float), idx

def train_epoch(model,features_mp16,loc_features_mp16,text_features_mp16, retrieval_indices,retrieval_indices_mp16, retrieval_similarities,retrieval_similarities_mp16, metadata, retrieval_tokens_table,retrieval_tokens_table_mp16,loc_features_100k, dataloader, optimizer,scheduler, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", unit="batch", dynamic_ncols=True)
    for features, loc_features, text_features, tokens, lat, lon, indices in pbar:
        features = features.to(device)
        loc_features = loc_features.to(device)
        text_features = text_features.to(device)
        tokens = tokens.to(device)
        optimizer.zero_grad()
        batch_retrieval_indices = torch.tensor(retrieval_indices[indices], dtype=torch.long, device=device)
        batch_retrieval_indices_mp16 = torch.tensor(retrieval_indices_mp16[indices], dtype=torch.long, device=device)
        batch_retrieval_similarities = torch.tensor(retrieval_similarities[indices], dtype=torch.float, device=device)
        batch_retrieval_similarities_mp16 = torch.tensor(retrieval_similarities_mp16[indices], dtype=torch.float, device=device)
        with autocast(dtype=torch.bfloat16):
            _, loss = model(features,loc_features,text_features,features_mp16,loc_features_mp16,text_features_mp16, batch_retrieval_indices,batch_retrieval_indices_mp16, batch_retrieval_similarities,batch_retrieval_similarities_mp16, metadata,
                        target_tokens=tokens, teacher_forcing=True, retrieval_tokens_table=retrieval_tokens_table,retrieval_tokens_table_mp16=retrieval_tokens_table_mp16,loc_features_100k=loc_features_100k)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.6f}"})
        
    scheduler.step()
    return total_loss / len(dataloader.dataset)

def forward_inference(model,features,loc_features,text_features,features_mp16,loc_features_mp16,text_features_mp16, retrieval_indices,retrieval_indices_mp16, retrieval_similarities,retrieval_similarities_mp16, metadata,tokens, retrieval_tokens_table,retrieval_tokens_table_mp16,loc_features_100k):
    pred_tokens = model(features,loc_features,text_features,features_mp16,loc_features_mp16,text_features_mp16, retrieval_indices,retrieval_indices_mp16, retrieval_similarities,retrieval_similarities_mp16, metadata,target_tokens=tokens, target_coords=None, teacher_forcing=False, retrieval_tokens_table=retrieval_tokens_table,retrieval_tokens_table_mp16=retrieval_tokens_table_mp16,loc_features_100k=loc_features_100k)
    return pred_tokens


def evaluate(model,features_mp16,loc_features_mp16,text_features_mp16, retrieval_indices,retrieval_indices_mp16, retrieval_similarities,retrieval_similarities_mp16, metadata, dataloader, device, s2_level, group_size, retrieval_tokens_table,retrieval_tokens_table_mp16,loc_features_100k):
    model.eval()
    all_preds = []  # will collect predicted coordinates as tensors ([lat, lon])
    all_trues = []  # will collect true coordinates as tensors ([lat, lon])
    with torch.no_grad():
        for features,loc_features,text_features, tokens, lat, lon, indices in tqdm(dataloader, desc="Evaluating", unit="batch"):
            features = features.to(device)
            loc_features = loc_features.to(device)
            text_features = text_features.to(device)
            # Create batch retrieval indices and similarities
            batch_retrieval_indices = torch.tensor(retrieval_indices[indices], dtype=torch.long, device=device)
            batch_retrieval_similarities = torch.tensor(retrieval_similarities[indices], dtype=torch.float, device=device)
            tokens = tokens.to(device)
            batch_retrieval_indices_mp16 = torch.tensor(retrieval_indices_mp16[indices], dtype=torch.long, device=device)
            batch_retrieval_similarities_mp16 = torch.tensor(retrieval_similarities_mp16[indices], dtype=torch.float, device=device)
            # Run inference (the returned tensor has shape (B, max_target_length))
            pred_grouped_tokens = forward_inference(
                model,features,loc_features,text_features,features_mp16,loc_features_mp16,text_features_mp16, batch_retrieval_indices,batch_retrieval_indices_mp16, batch_retrieval_similarities,batch_retrieval_similarities_mp16, metadata, tokens, retrieval_tokens_table,retrieval_tokens_table_mp16,loc_features_100k
            )
            
            # Process each sample in the batch independently
            batch_size = pred_grouped_tokens.size(0)
            for i in range(batch_size):
                # Get the grouped token predictions for this sample as a list
                token_list = pred_grouped_tokens[i].tolist()
                # Optionally, you may ungroup the tokens if needed for debugging
                full_tokens = ungroup_s2_tokens(token_list, group_size=group_size)
                token_list = full_tokens
                # Convert the (grouped) tokens to predicted coordinates
                pred_lat, pred_lon = s2_tokens_to_latlng(token_list, s2_level)
                all_preds.append(torch.tensor([pred_lat, pred_lon], dtype=torch.float))

            # Process the true coordinates for this batch (assumes lat, lon are batched tensors)
            true_coords = torch.stack([lat, lon], dim=1).cpu()  # shape: (B, 2)

            all_trues.append(true_coords)
    
    # Concatenate all predictions and true coordinates over batches.
    all_preds = torch.stack(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)
    metrics = compute_metrics(all_preds, all_trues)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train GeoTransformerModel with evaluation on YFCC4K and im2gps3k")
    # Training data.
    parser.add_argument('--annotations', type=str,
                        help='Path to training CSV (IMG_ID, LAT, LON, ...)', 
                        default="data/MP16_Pro_filtered.csv")
    parser.add_argument('--features_file', type=str, 
                        help='Path to precomputed training features (e.g., train_embeds.npy)', 
                        default="modules_all/faiss_outputs/mp16_raw_image_feats.npy")
    parser.add_argument('--loc_features_file',type=str,
                        help='Path to precomputed training location features (e.g., train_loc_feats.npy)', 
                        default="modules_all/faiss_outputs/mp16_raw_lochead_feats.npy")
    parser.add_argument('--text_features_file',type=str,
                        help='Path to precomputed training text features (e.g., train_text_feats.npy)', 
                        default="modules_all/faiss_outputs/mp16_raw_text_feats.npy")
    parser.add_argument('--tokens_file_mp16', type=str, 
                        help='Path to precomputed S2 tokens for training (e.g., s2_tokens_mp16.npy)', 
                        default="data/S2/s2_tokens_mp16_grouped_2.npy")
    parser.add_argument('--tokens_file_100k', type=str, 
                        help='Path to precomputed S2 tokens for training (e.g., s2_tokens_100k.npy)', 
                        default="data/S2/s2_tokens_100k_grouped_2.npy")
    parser.add_argument('--retrieval_indices_file', type=str, 
                        help='Path to precomputed training retrieval indices (e.g., I_train.npy)', 
                        default="modules_all/gallery_retrieval_100k/I_train_100K.npy")
    parser.add_argument('--retrieval_similarities_file', type=str,
                        help='Path to precomputed training retrieval similarities (e.g., D_train.npy)', 
                        default="modules_all/gallery_retrieval_100k/D_train_100K.npy")

    parser.add_argument('--retrieval_indices_file_mp16', type=str, 
                        help='Path to precomputed training retrieval indices (e.g., I_train.npy)', 
                        default="modules_all/faiss_outputs/I_mp16.npy")
    parser.add_argument('--retrieval_similarities_file_mp16', type=str,
                        help='Path to precomputed training retrieval similarities (e.g., D_train.npy)', 
                        default="modules_all/faiss_outputs/D_mp16.npy")


    parser.add_argument('--loc_features_100k_file', type=str,
                        help='Path to precomputed training location features (e.g., train_loc_feats.npy)', 
                        default="modules_all/gallery_retrieval_100k/gps_gallery_loc.npy")


    # parser.add_argument('--gallery_text_embeddings', type=str,
                        # help='Path to precomputed gallery text embeddings (e.g., gallery_text_embeddings.npy)', 
                        # default="data/mp16_text_embeddings.npy")
                                           

    parser.add_argument('--metadata_file', type=str,
                        help='Path to metadata.npy (shape: [N_meta, 2])', 
                        default="modules_all/faiss_outputs/mp16_meta.npy")
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--sim_dim', type=int, default=512, help='Common embedding dimension (equal to d_model)')
    parser.add_argument('--transformer_output', type=str, default="geo_transformer_model_GC_100_MP16_IMf_full_bigger_d_10_text_loss_1e-4_256_grouped_2_without100k.pth")
    parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency (in epochs)')
    # S2 level used for tokenization.
    parser.add_argument('--s2_level', type=int, default=20, help='S2 level (full token sequence length = level+1)')
    
    # Evaluation data for YFCC4K.
    parser.add_argument('--eval_yfcc_annotations', type=str,
                        help='Path to YFCC4K test CSV annotations', 
                        default="data/yfcc4k_places.csv")

    parser.add_argument('--eval_yfcc_features', type=str, 
                        help='Path to precomputed YFCC4K features (e.g., yfcc_features.npy)', 
                        default="modules_all/faiss_outputs/yfcc_raw_image_feats.npy")
    parser.add_argument('--eval_yfcc_loc_features', type=str, 
                        help='Path to precomputed YFCC4K location features (e.g., yfcc_loc_feats.npy)', 
                        default="modules_all/faiss_outputs/yfcc_raw_lochead_feats.npy")
    parser.add_argument('--eval_yfcc_text_features', type=str, 
                        help='Path to precomputed YFCC4K text features (e.g., yfcc_text_feats.npy)', 
                        default="modules_all/faiss_outputs/yfcc_raw_text_feats.npy")

    parser.add_argument('--eval_yfcc_tokens', type=str,
                        help='Path to precomputed S2 tokens for YFCC4K (e.g., s2_tokens_yfcc4k.npy)', 
                        default="data/S2/s2_tokens_yfcc4k_grouped_2.npy")
    parser.add_argument('--eval_yfcc_retrieval_indices_file', type=str, 
                        help='Path to YFCC4K retrieval indices (e.g., I_test_yfcc4k.npy)', 
                        default="modules_all/gallery_retrieval_100k/I_yfcc_100K.npy")
    parser.add_argument('--eval_yfcc_retrieval_similarities_file', type=str, 
                        help='Path to YFCC4K retrieval similarities (e.g., D_test_yfcc4k.npy)', 
                        default="modules_all/gallery_retrieval_100k/D_yfcc_100K.npy")
    
    parser.add_argument('--eval_yfcc_retrieval_indices_file_mp16', type=str, 
                        help='Path to YFCC4K retrieval indices (e.g., I_test_yfcc4k.npy)', 
                        default="modules_all/faiss_outputs/I_yfcc.npy")
    parser.add_argument('--eval_yfcc_retrieval_similarities_file_mp16', type=str, 
                        help='Path to YFCC4K retrieval similarities (e.g., D_test_yfcc4k.npy)', 
                        default="modules_all/faiss_outputs/D_yfcc.npy")
    
    # Evaluation data for im2gps3k.
    parser.add_argument('--eval_im2gps_annotations', type=str,
                        help='Path to im2gps3k test CSV annotations', 
                        default="data/im2gps3k_places365.csv")

    parser.add_argument('--eval_im2gps_features', type=str, 
                        help='Path to precomputed im2gps3k features (e.g., im2gps_features.npy)', 
                        default="modules_all/faiss_outputs/im2gps_raw_image_feats.npy")
    parser.add_argument('--eval_im2gps_loc_features', type=str, 
                        help='Path to precomputed im2gps3k location features (e.g., im2gps_loc_feats.npy)', 
                        default="modules_all/faiss_outputs/im2gps_raw_lochead_feats.npy")
    parser.add_argument('--eval_im2gps_text_features', type=str, 
                        help='Path to precomputed im2gps3k text features (e.g., im2gps_text_feats.npy)', 
                        default="modules_all/faiss_outputs/im2gps_raw_text_feats.npy")

    parser.add_argument('--eval_im2gps_tokens', type=str, 
                        help='Path to precomputed S2 tokens for im2gps3k (e.g., s2_tokens_im2gps3k.npy)', 
                        default="data/S2/s2_tokens_im2gps3k_grouped_2.npy")
    parser.add_argument('--eval_im2gps_retrieval_indices_file', type=str, 
                        help='Path to im2gps3k retrieval indices (e.g., I_test_im2gps3k.npy)', 
                        default="modules_all/gallery_retrieval_100k/I_im2gps_100K.npy")
    parser.add_argument('--eval_im2gps_retrieval_similarities_file', type=str, 
                        help='Path to im2gps3k retrieval similarities (e.g., D_test_im2gps3k.npy)', 
                        default="modules_all/gallery_retrieval_100k/D_im2gps_100K.npy")

    parser.add_argument('--eval_im2gps_retrieval_indices_file_mp16', type=str, 
                        help='Path to im2gps3k retrieval indices (e.g., I_test_im2gps3k.npy)', 
                        default="modules_all/faiss_outputs/I_im2gps.npy")
    parser.add_argument('--eval_im2gps_retrieval_similarities_file_mp16', type=str, 
                        help='Path to im2gps3k retrieval similarities (e.g., D_test_im2gps3k.npy)', 
                        default="modules_all/faiss_outputs/D_im2gps.npy")

    parser.add_argument('--group_size', type=int, default=2, help='Grouping size for tokens')
    parser.add_argument('--max_seq_length', type=int, default=11, help='Length of grouped token sequence')




    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create subset of training dataset (first 2000 samples) for evaluation.

    # Load training dataset.
    train_dataset = PrecomputedFeatureTokenDataset(args.annotations, args.features_file, args.loc_features_file, args.text_features_file, args.tokens_file_mp16)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    retrieval_indices = np.load(args.retrieval_indices_file)  # (N_train, k)
    retreival_indices_mp16 = np.load(args.retrieval_indices_file_mp16)  # (N_train, k)
    retrieval_similarities = np.load(args.retrieval_similarities_file)  # (N_train, k)
    retrieval_similarities_mp16 = np.load(args.retrieval_similarities_file_mp16)  # (N_train, k)
    metadata = np.load(args.metadata_file)  # (N_meta, 2)
    features = np.load(args.features_file)  # (N_train, 768)
    loc_features = np.load(args.loc_features_file)  # (N_train, 768)
    text_features = np.load(args.text_features_file)  # (N_train, 768)


    loc_features_100k = np.load(args.loc_features_100k_file)  # (N_train, 768)
    train_subset = Subset(train_dataset, list(range(min(len(train_dataset), len(train_dataset)))))
    train_subset_loader = DataLoader(train_subset,batch_size=args.batch_size , shuffle=False, num_workers=1)
    retrieval_indices_subset = retrieval_indices[:min((len(train_dataset)), len(train_dataset))]
    retrieval_similarities_subset = retrieval_similarities[:min((len(train_dataset)), len(train_dataset))]
    retrieval_indices_subset_mp16 = retreival_indices_mp16[:min((len(train_dataset)), len(train_dataset))]
    retrieval_similarities_subset_mp16 = retrieval_similarities_mp16[:min((len(train_dataset)), len(train_dataset))]
    
    # Create subset of training dataset (first 2000 samples) for evaluation.
    train_subset_t = Subset(train_dataset, list(range(min(len(train_dataset), len(train_dataset)))))
    train_subset_loader_t = DataLoader(train_subset_t, batch_size=args.batch_size, shuffle=False, num_workers=1)
    retrieval_indices_subset_t = retrieval_indices[:min(len(train_dataset), len(train_dataset))]
    retrieval_similarities_subset_t = retrieval_similarities[:min(len(train_dataset),  len(train_dataset))]
    retrieval_indices_subset_mp16_t = retreival_indices_mp16[:min(len(train_dataset), len(train_dataset))]
    retrieval_similarities_subset_mp16_t = retrieval_similarities_mp16[:min(len(train_dataset), len(train_dataset))]


    model = GeoTransformerModelS2(device, d_model=args.sim_dim, transformer_layers=10, nhead=8, ff_dim=args.sim_dim*2,
                                  dropout=0.1, s2_level=args.s2_level, group_size=args.group_size, max_seq_length=args.max_seq_length).to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    import torch.optim as optim

    # Set initial learning rate to 1e-3
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Optimizer with weight decay
    LR = args.lr* (0.9**26)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    # lr = 0.000026
    # Create a StepLR scheduler that reduces lr by a factor of gamma every 'step_size' epochs.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.90)

    eval_yfcc_dataset = PrecomputedFeatureTokenDataset(args.eval_yfcc_annotations, args.eval_yfcc_features, args.eval_yfcc_loc_features, args.eval_yfcc_text_features, args.eval_yfcc_tokens)
    eval_yfcc_loader = DataLoader(eval_yfcc_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    eval_yfcc_retrieval_indices = np.load(args.eval_yfcc_retrieval_indices_file)  # (N_yfcc, k)
    eval_yfcc_retrieval_similarities = np.load(args.eval_yfcc_retrieval_similarities_file)  # (N_yfcc, k)
    eval_yfcc_retrieval_indices_mp16 = np.load(args.eval_yfcc_retrieval_indices_file_mp16)  # (N_yfcc, k)
    eval_yfcc_retrieval_similarities_mp16 = np.load(args.eval_yfcc_retrieval_similarities_file_mp16)  # (N_yfcc, k)
    
    eval_im2gps_dataset = PrecomputedFeatureTokenDataset(args.eval_im2gps_annotations, args.eval_im2gps_features, args.eval_im2gps_loc_features, args.eval_im2gps_text_features, args.eval_im2gps_tokens)
    eval_im2gps_loader = DataLoader(eval_im2gps_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    eval_im2gps_retrieval_indices = np.load(args.eval_im2gps_retrieval_indices_file)  # (N_im2gps, k)
    eval_im2gps_retrieval_similarities = np.load(args.eval_im2gps_retrieval_similarities_file)  # (N_im2gps, k)
    eval_im2gps_retrieval_indices_mp16 = np.load(args.eval_im2gps_retrieval_indices_file_mp16)  # (N_im2gps, k)
    eval_im2gps_retrieval_similarities_mp16 = np.load(args.eval_im2gps_retrieval_similarities_file_mp16)  # (N_im2gps, k)
    
    retrieval_tokens_table = torch.tensor(np.load(args.tokens_file_100k), dtype=torch.long, device=device)
    retrieval_tokens_table_mp16 = torch.tensor(np.load(args.tokens_file_mp16), dtype=torch.long, device=device)
   
    Epoch = 0
    path = args.transformer_output.split(".")[0] + f"_epoch_{Epoch}.pth"

    model.load_state_dict(torch.load(path))

    for epoch in range(Epoch,args.epochs):
        train_loss = train_epoch(model,features,loc_features,text_features, retrieval_indices_subset_t,retrieval_indices_subset_mp16_t , retrieval_similarities_subset_t,retrieval_similarities_subset_mp16_t, metadata,retrieval_tokens_table,retrieval_tokens_table_mp16,loc_features_100k,train_subset_loader_t, optimizer,scheduler, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {train_loss:.4f}")
        
        if (epoch + 1) % args.eval_freq == 0:

            eval_yfcc_metrics = evaluate(model,features,loc_features,text_features, eval_yfcc_retrieval_indices,eval_yfcc_retrieval_indices_mp16, eval_yfcc_retrieval_similarities,eval_yfcc_retrieval_similarities_mp16, metadata,
                                     eval_yfcc_loader, device, s2_level=args.s2_level, group_size=args.group_size,retrieval_tokens_table=retrieval_tokens_table,retrieval_tokens_table_mp16 = retrieval_tokens_table_mp16,loc_features_100k=loc_features_100k)
            print(f"Epoch {epoch+1}/{args.epochs}, YFCC4K Evaluation Metrics: {eval_yfcc_metrics}")
        
            eval_im2gps_metrics = evaluate(model,features,loc_features,text_features, eval_im2gps_retrieval_indices,eval_im2gps_retrieval_indices_mp16, eval_im2gps_retrieval_similarities,eval_im2gps_retrieval_similarities_mp16, metadata,
                                       eval_im2gps_loader, device, s2_level=args.s2_level, group_size=args.group_size,retrieval_tokens_table=retrieval_tokens_table,retrieval_tokens_table_mp16 = retrieval_tokens_table_mp16,loc_features_100k=loc_features_100k)
            print(f"Epoch {epoch+1}/{args.epochs}, im2gps3k Evaluation Metrics: {eval_im2gps_metrics}")
        path = args.transformer_output.split(".")[0] + f"_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), path)
        print(f"Model saved as {path}")



if __name__ == "__main__":
    main()
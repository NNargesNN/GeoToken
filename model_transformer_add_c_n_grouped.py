import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from s2_token_utils import latlng_to_s2_tokens, group_s2_tokens, ungroup_s2_tokens, s2_tokens_to_latlng
import numpy as np

tokens_generated_during_training = None
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class GeoTransformerModelS2(nn.Module):
    def __init__(self, device, d_model=64, transformer_layers=4, nhead=8, ff_dim=256, dropout=0.1,
                 s2_level=20, group_size=5, max_seq_length=None, num_neighbors=5,token_vocab_sizes=None):
        """
        Args:
            device: Torch device.
            d_model: Transformer token dimension.
            transformer_layers: Number of encoder/decoder layers.
            nhead: Number of attention heads.
            ff_dim: Feedforward dimension.
            dropout: Dropout rate.
            s2_level: S2 level for token conversion (full token sequence length = s2_level + 1).
            group_size: Grouping size for S2 tokens.
            max_seq_length: Length of the grouped token sequence for the target.
                Default: 1 + (s2_level // group_size) (s2_level must be divisible by group_size).
            num_neighbors: Number of retrieved neighbors (k).
        """

        super(GeoTransformerModelS2, self).__init__()
        self.device = device
        self.d_model = d_model
        if max_seq_length is None:
            if s2_level % group_size != 0:
                raise ValueError("s2_level must be divisible by group_size")
            self.max_seq_length = 1 + (s2_level // group_size)  # Includes the face token.
        else:
            self.max_seq_length = max_seq_length
        # For decoding we predict only the target tokens (positions 1 ... max_seq_length - 1).
        self.max_target_length = self.max_seq_length - 1

        # Define vocabulary sizes.
        # Position 0 (face) has 6 possibilities; positions 1...max_seq_length-1 each have 4^(group_size) possibilities.
        full_vocab_sizes = [6] + [4 ** group_size] * (self.max_seq_length - 1)
        # For prediction we use positions 1 .. end.
        self.target_token_vocab_sizes = full_vocab_sizes
        if token_vocab_sizes is not None:
            self.target_token_vocab_sizes = token_vocab_sizes


        self.position_weights = torch.linspace(
            start=2.0,
            end=1.0,
            steps=self.max_seq_length,
        ).to(device)  # shape: (max_seq_length,)

        # added for classification head:

        
        self.output_layers = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for vocab_size in self.target_token_vocab_sizes
        ])
        # Compute offsets for target tokens.
        self.token_offsets = []
        current_offset = 0
        for vocab_size in self.target_token_vocab_sizes:
            self.token_offsets.append(current_offset)
            current_offset += vocab_size

        # Embedding for target tokens (indices are assumed to be in [0, vocab_size) for each position).
        total_target_vocab = sum(self.target_token_vocab_sizes)
        # print("total_target_vocab: ", total_target_vocab)
        self.embedding = nn.Embedding(total_target_vocab, d_model)

        # Learnable start token embedding (separate from the token embedding).
        self.start_token_embedding = nn.Parameter(torch.randn(1, d_model))
        # Learnable CLS token for the encoder.
        self.cls_token = nn.Parameter(torch.randn(1, d_model))
        
        # Project precomputed image features (768-d) to d_model.
        self.image_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.loc_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.retrieval_embedding = nn.Embedding(total_target_vocab, d_model)
        self.similarity_embedding = nn.Embedding(11, d_model)

        # Encoder: process a source sequence composed of the image and retrieval information.
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.encoder_pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=500)  # small max_len
        
        # For retrieval: project metadata coordinates (2-d) to d_model.
        self.metadata_proj = nn.Linear(2, d_model)
        
        # Coordinate prediction head (predicts lat,lon from the encoder output, e.g. from the image token).
        self.coord_head = nn.Linear(d_model, 2)
        
        # Decoder: autoregressively generates target token embeddings.
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=transformer_layers)
        self.decoder_pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_seq_length+1)
        self.targt_coord_loss_scale = 1e-3

    def forward(self, image_features,loc_features,text_features,features_mp16,loc_features_mp16,text_features_mp16, retrieval_indices,retrieval_indices_mp16, retrieval_similarities,retrieval_similarities_mp16, metadata,
                target_tokens=None,target_coords=None, teacher_forcing=True, retrieval_tokens_table=None,retrieval_tokens_table_mp16=None,loc_features_100k=None):
        """

        Args:
            image_features: Tensor (batch, 768).
            loc_features: Tensor (batch, 768).
            text_features: Tensor (batch, 768).
            features_mp16: Tensor (batch, 768).
            loc_features_mp16: Tensor (batch, 768).
            text_features_mp16: Tensor (batch, 768).
            retrieval_indices: LongTensor (batch, num_neighbors) – indices into metadata.
            retrieval_similarities: FloatTensor (batch, num_neighbors).
            metadata: NumPy array (N_meta, 2).
            target_tokens: Ground-truth grouped token sequences (batch, max_seq_length).
            teacher_forcing: If True, use teacher forcing.
        Returns:
            If teacher_forcing: (None, loss) averaged over time steps.
            Otherwise: predicted grouped token sequences (batch, max_seq_length).
        """
        global tokens_generated_during_training
        
        batch_size = image_features.size(0)
        # print("batch_size,",batch_size)
        if not torch.is_tensor(metadata):
            metadata = torch.tensor(metadata, dtype=torch.float, device=image_features.device)
        if retrieval_indices.dim() == 1:
            retrieval_indices = retrieval_indices.reshape(1, -1)
        if retrieval_indices_mp16.dim() == 1:
            retrieval_indices_mp16 = retrieval_indices_mp16.reshape(1, -1)


        if retrieval_indices.dim() == 1:
            retrieval_indices = retrieval_indices.unsqueeze(0)
        if retrieval_similarities.dim() == 1:
            retrieval_similarities = retrieval_similarities.unsqueeze(0)

        if retrieval_indices_mp16.dim() == 1:
            retrieval_indices_mp16 = retrieval_indices_mp16.unsqueeze(0)
        if retrieval_similarities_mp16.dim() == 1:
            retrieval_similarities_mp16 = retrieval_similarities_mp16.unsqueeze(0)
        # if teacher_forcing:
        # target_coords = metadata[retrieval_indices[:, 0]].reshape(batch_size, 2)  # (batch, 2)
        retrieval_indices = retrieval_indices[:,:5]
        retrieval_similarities = retrieval_similarities[:,:5]
        if teacher_forcing:
            retrieval_indices_mp16 = retrieval_indices_mp16[:,1:16]
            retrieval_similarities_mp16 = retrieval_similarities_mp16[:,1:16]
        else:   
            retrieval_indices_mp16 = retrieval_indices_mp16[:,0:15]
            retrieval_similarities_mp16 = retrieval_similarities_mp16[:,0:15]
        
        
            
        batch_size = image_features.size(0)

        # === Encoder: image + retrieval ===
        # Project image features.
        img_embeds = self.image_proj(image_features)  # (batch, d_model)
        img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
        loc_embeds = self.loc_proj(loc_features)  # (batch, d_model)
        loc_embeds = loc_embeds / loc_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = self.text_proj(text_features)  # (batch, d_model)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)


        k_mp16 = retrieval_indices_mp16.size(1)  # number of neighbors to process
        k_100k = retrieval_indices.size(1)  # number of neighbors to process
        
        neighbor_seq_list = []  # will hold each neighbor's full sequence of token embeddings
        neighbor_seq_list_mp16 = []  # will hold each neighbor's full sequence of token embeddings
        neighbor_image_embeds_list = []  # will hold each neighbor's image embedding
        for j in range(k_mp16):
           
            idx_j_mp16 = retrieval_indices_mp16[:, j]   # (batch,)

            neighbor_tokens_mp16 = retrieval_tokens_table_mp16[idx_j_mp16]  # (batch, self.max_seq_length)                    
  
            idx_j_mp16_cpu = idx_j_mp16.cpu().numpy()  # move indices to CPU for numpy indexing


            batch_features_mp16 = torch.from_numpy(features_mp16[idx_j_mp16_cpu]).float().to(self.device)
            neighbor_image_embeds = self.image_proj(batch_features_mp16)
            neighbor_image_embeds = neighbor_image_embeds / neighbor_image_embeds.norm(p=2, dim=-1, keepdim=True)


            batch_loc_features_mp16 = torch.from_numpy(loc_features_mp16[idx_j_mp16_cpu]).float().to(self.device)
            neighbor_loc_embeds = self.loc_proj(batch_loc_features_mp16)  # (batch, d_model)
            neighbor_loc_embeds = neighbor_loc_embeds / neighbor_loc_embeds.norm(p=2, dim=-1, keepdim=True)


            batch_text_features_mp16 = torch.from_numpy(text_features_mp16[idx_j_mp16_cpu]).float().to(self.device)
            neighbor_text_embeds = self.text_proj(batch_text_features_mp16)  # (batch, d_model)
            neighbor_text_embeds = neighbor_text_embeds / neighbor_text_embeds.norm(p=2, dim=-1, keepdim=True)

            # emb_tokens_list = []
            emb_tokens_list_mp16 = []
            for pos in range(neighbor_tokens_mp16.size(1)):
                # Shift token by position-specific offset.

                token_ids_mp16 = neighbor_tokens_mp16[:, pos] + self.token_offsets[pos]

                emb_mp16 = self.retrieval_embedding(token_ids_mp16)  # (batch, d_model)
                # emb_tokens_list.append(emb)
                emb_tokens_list_mp16.append(emb_mp16)


            # stack the grouped-token embeddings:
            neighbor_seq_mp16 = torch.stack(emb_tokens_list_mp16, dim=1)  # (batch, seq_len, d_model)
            # we want the image embedding first: concat it as a new time-step:
            neighbor_seq_mp16 = torch.cat(
                [neighbor_image_embeds.unsqueeze(1),neighbor_loc_embeds.unsqueeze(1),neighbor_text_embeds.unsqueeze(1), neighbor_seq_mp16],
                dim=1
            )  # now (batch, seq_len+1, d_model)

            neighbor_seq_list_mp16.append(neighbor_seq_mp16)
        



     
        retrieval_tokens_tensor_mp16 = torch.stack(neighbor_seq_list_mp16, dim=1)

        retrieval_tokens_flat_mp16 = retrieval_tokens_tensor_mp16.view(batch_size, -1, self.d_model)

        # --- Build encoder input sequence.
        # Start with a CLS token and the image token.
        cls_tokens = self.cls_token.expand(batch_size, -1).unsqueeze(0)  # (1, batch, d_model)
        img_embeds_enc = img_embeds.unsqueeze(0)  # (1, batch, d_model)
        loc_embeds_enc = loc_embeds.unsqueeze(0)  # (1, batch, d_model)
        text_embeds_enc = text_embeds.unsqueeze(0)  # (1, batch, d_model)

        retrieval_tokens_enc_mp16 = retrieval_tokens_flat_mp16.transpose(0, 1)  # (k*self.max_seq_length, batch, d_model)
  
        encoder_inputs = torch.cat([cls_tokens, img_embeds_enc, loc_embeds_enc,text_embeds_enc,retrieval_tokens_enc_mp16], dim=0)

        encoder_inputs = self.encoder_pos_encoder(encoder_inputs)
        memory = self.encoder(encoder_inputs)  

        coord_loss = 0.0

        if teacher_forcing:
           
            dec_inputs = []
            # t = 0: use the learnable start token.
            start_token = self.start_token_embedding.expand(batch_size, -1)  # (batch, d_model)
            dec_inputs.append(start_token)
            # t = 1 ... max_seq_length-1: use teacher forcing embeddings.

            for t in range(self.max_seq_length - 1):

                token_ids = target_tokens[:, t] + self.token_offsets[t]
                emb = self.embedding(token_ids)  # (batch, d_model)
                dec_inputs.append(emb)
            dec_inputs = torch.stack(dec_inputs, dim=0)
            dec_inputs = self.decoder_pos_encoder(dec_inputs)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_inputs.size(0)).to(image_features.device)
            decoder_output = self.decoder(dec_inputs, memory, tgt_mask=tgt_mask)  # (max_seq_length, batch, d_model)
            

            normalizer = self.position_weights.sum()


            token_loss = 0.0
            generated_tokens = []
            criterion = nn.CrossEntropyLoss(reduction='none')
            for t in range(self.max_seq_length):
                logits = self.output_layers[t](decoder_output[t])        # [batch, vocab]
                loss_t = nn.functional.cross_entropy(logits, target_tokens[:, t], reduction='mean')
                # weight it according to position:
                token_loss += self.position_weights[t] * loss_t

            tokens_generated_during_training = generated_tokens
            # token_loss = token_loss / self.max_seq_length
            token_loss = token_loss / normalizer

            total_loss = token_loss

            return None, total_loss
        else:
            # Inference: autoregressive decoding.
            dec_inputs = [self.start_token_embedding.expand(batch_size, -1)]

            # print("EVALUATE")
            generated_tokens = []  # to collect predicted tokens for each target position.
            for t in range(self.max_seq_length):
                current_seq = torch.stack(dec_inputs, dim=0)  # (current_seq_len, batch, d_model)
                current_seq = self.decoder_pos_encoder(current_seq)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_seq.size(0)).to(image_features.device)
                dec_out = self.decoder(current_seq, memory, tgt_mask=tgt_mask)
                last_output = dec_out[-1]  # (batch, d_model)
                logits = self.output_layers[t](last_output)  # (batch, vocab_size for this position)
                # print(f"{t}: logtis: {logits}")
                pred_token = torch.argmax(logits, dim=-1)  # (batch,)
                generated_tokens.append(pred_token)
                # Append embedding of predicted token (with appropriate offset) as next input.
                full_token = pred_token + self.token_offsets[t]
                dec_inputs.append(self.embedding(full_token))
            # Stack along time dimension to produce (batch, max_target_length)
            pred_tokens = torch.stack(generated_tokens, dim=1)

            return pred_tokens
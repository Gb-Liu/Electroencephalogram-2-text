import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=1024, dropout=0.6):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        #self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        projected = self.projection(x)
        #x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x

class BrainTranslator(nn.Module):
    def __init__(self, bart, in_feature=1024, decoder_embedding_size=1024, additional_encoder_nhead=8,
                 additional_encoder_dim_feedforward=2048):
        super(BrainTranslator, self).__init__()

        # Embedded EEG raw features
        self.hidden_dim = in_feature
        self.conv_module = ConvolutionModule(in_channels=56,out_channels=56,output_dim=self.hidden_dim)
        self.fc = ProjectionHead(embedding_dim=self.hidden_dim, projection_dim=self.hidden_dim, dropout=0.6)  # nn.Linear(in_feature, in_feature)

        # Brain transformer encoder
        self.num_layers = 8
        self.pos_embedding = nn.Parameter(torch.randn(1, 56, self.hidden_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,  nhead=additional_encoder_nhead,
            dim_feedforward=additional_encoder_dim_feedforward,   dropout=0.6,
            activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,  num_layers=self.num_layers)
        self.layernorm_embedding = nn.LayerNorm(decoder_embedding_size,  eps=1e-05)
        self.brain_projection = ProjectionHead(embedding_dim=self.hidden_dim,
            projection_dim=decoder_embedding_size,  dropout=0.6)
        # BART
        self.bart = bart

    def freeze_pretrained_bart(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
            if ('bart' in name):
                param.requires_grad = False

    def freeze_pretrained_brain(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            if ('bart' in name):
                param.requires_grad = True

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert,
                target_ids_batch_converted, lenghts_words, word_contents_batch, word_contents_attn_batch,
                stepone, subject_batch, device, features=False):
        if input_embeddings_batch.size(1) != self.pos_embedding.size(1):
            print(f"Warning: Input sequence length {input_embeddings_batch.size(1)} "
                  f"does not match position encoding length {self.pos_embedding.size(1)}")

        feature_embedding = self.conv_module(input_embeddings_batch, device)

        if len(feature_embedding.shape) == 2:
            feature_embedding = torch.unsqueeze(feature_embedding, 0)
        encoded_embedding = self.fc(feature_embedding)  # encoded_embedding.shape[4, 56, 1024]
        brain_embedding = encoded_embedding + self.pos_embedding
        brain_embedding = self.encoder(brain_embedding, src_key_padding_mask=input_masks_invert)
        brain_embedding = self.brain_projection(brain_embedding)
        brain_embedding = self.layernorm_embedding(brain_embedding)

        if stepone == True:
            words_embedding = self.bart.model.encoder.embed_tokens(word_contents_batch)
            loss = nn.MSELoss()
            return loss(brain_embedding, words_embedding)
        else:
            if brain_embedding.size(1) != input_masks_batch.size(1):
                # print(f"Adjusting input_masks_batch from {input_masks_batch.size(1)} to {brain_embedding.size(1)}")
                input_masks_batch = input_masks_batch[:, :brain_embedding.size(1)]
            max_length = self.bart.config.max_position_embeddings
            if brain_embedding.size(1) > max_length:
                print(f"Truncating sequence from {brain_embedding.size(1)} to {max_length}")
                brain_embedding = brain_embedding[:, :max_length, :]
                input_masks_batch = input_masks_batch[:, :max_length]
            out = self.bart(inputs_embeds=brain_embedding, attention_mask=input_masks_batch,
                            return_dict=True, labels=target_ids_batch_converted)
            if features == True:
                return out.logits, brain_embedding

            return out.logits

class ConvolutionModule(nn.Module):
    def __init__(self, in_channels=56, out_channels=56, output_dim=840, target_length=56):
        super().__init__()

        self.num_kernels = 3

        self.conv2 = nn.Conv2d( in_channels=in_channels, out_channels=out_channels,
            kernel_size=(33, 2), stride=(2, 2), padding=(15, 0), dilation=1 )
        self.conv3 = nn.Conv2d( in_channels=in_channels, out_channels=out_channels,
            kernel_size=(9, 2), stride=(2, 2), padding=(3, 0), dilation=1 )
        self.conv4 = nn.Conv2d( in_channels=in_channels, out_channels=out_channels,
            kernel_size=(5, 2), stride=(2, 2), padding=(1, 0), dilation=1 )
        
        self.conv = nn.Conv2d( in_channels=in_channels, out_channels=out_channels, kernel_size=(4, 5), stride=(2, 1), padding=(10, 0), dilation= 1)
        
        self.flatten = nn.Flatten(start_dim=2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(840, 1024)

    def forward(self, x, device):
      
        out2 = self.conv2(x)  # (batch, 56, 52, 12)
        out3 = self.conv3(x)  # (batch, 56, 52, 12)
        out4 = self.conv4(x)  # (batch, 56, 52, 12)

        out2 = self.conv(self.relu(out2))  #[8, 56, 35, 8]
        out3 = self.conv(self.relu(out3))  
        out4 = self.conv(self.relu(out4))

        flat2 = self.flatten(out2)  
        flat3 = self.flatten(out3)  
        flat4 = self.flatten(out4) 


        combined = torch.cat([flat2, flat3, flat4], dim=2)
        combined = self.fc(combined)

        #output = self.relu(self.bn(combined))  
        output = self.bn(combined)

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Elements import Elements

class TransformerMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate):
        super(TransformerMLP, self).__init__()
        self.feedforward = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LazyLinear(embed_dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, X):
        return self.feedforward(X)



class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = TransformerMLP(embed_dim, hidden_dim, dropout_rate) 

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        attn_output, attn_weights = self.attention(out, out, out, need_weights=True)
        x = identity + attn_output

        identity = x
        out = self.norm2(x)
        out = self.mlp(out)
        x = identity + out

        return x, attn_weights

class VisionTransformer(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, num_heads, num_classes, num_layers, hidden_dim, dropout_rate, in_channels=1):
        super(VisionTransformer, self).__init__()
        num_patches = input_size // patch_size
        self.num_patches = num_patches 
        self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout_rate) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.fc = nn.Linear(embed_dim, num_classes)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.patch_embed(x) 
        x = x.transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x + self.pos_embed)

        for i, block in enumerate(self.transformer_blocks):
            x, _ = block(x)

        x = self.norm(x)
        cls_token_representation = x[:, 0]
        x = self.fc(cls_token_representation)
        x = self.sigmoid(x)

        return x

class LT_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super(LT_Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(dropout_rate)

    def forward(self, img_tokens, lbl_tokens):
        B, N_img, D = img_tokens.shape
        _, N_lbl, _ = lbl_tokens.shape
        N_total = N_img + N_lbl

        tokens = torch.cat((img_tokens, lbl_tokens), dim=1)

        qkv = self.qkv_proj(tokens).chunk(3, dim=-1) # Tuple of 3 tensors: (B, N_total, D) each
        q, k, v = map(lambda t: t.reshape(B, N_total, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        q_img, q_lbl = q[:, :, :N_img, :], q[:, :, N_img:, :]
        k_img, k_lbl = k[:, :, :N_img, :], k[:, :, N_img:, :] 
        v_img, v_lbl = v[:, :, :N_img, :], v[:, :, N_img:, :] 

        attn_scores_img = (q_img @ k_img.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_probs_img = attn_scores_img.softmax(dim=-1)
        attn_probs_img = self.attn_dropout(attn_probs_img)
        attn_output_img = (attn_probs_img @ v_img).transpose(1, 2).reshape(B, N_img, D)

        attn_scores_lbl = (q_lbl @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_probs_lbl = attn_scores_lbl.softmax(dim=-1)
        attn_probs_lbl = self.attn_dropout(attn_probs_lbl)
        attn_output_lbl = (attn_probs_lbl @ v).transpose(1, 2).reshape(B, N_lbl, D)

        # Combine outputs (apply output projection separately or together? Paper implies FFN after attn)
        # Let's follow Fig 1: apply residual, then Norm, then FFN in the block
        # The attention mechanism itself just produces the weighted value vectors.
        # We apply the final projection to the outputs of the attention mechanism.
        output_img = self.out_dropout(self.out_proj(attn_output_img))
        output_lbl = self.out_dropout(self.out_proj(attn_output_lbl))

        return output_img, output_lbl


# --- New LT Transformer Block (Uses LT_Attention) ---
class LT_TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_rate):
        super(LT_TransformerBlock, self).__init__()
        self.norm1_img = nn.LayerNorm(embed_dim)
        self.norm1_lbl = nn.LayerNorm(embed_dim)
        self.attention = LT_Attention(embed_dim, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = TransformerMLP(embed_dim, hidden_dim, dropout_rate)

    def forward(self, img_tokens, lbl_tokens):
        img_identity = img_tokens
        lbl_identity = lbl_tokens

        img_norm1 = self.norm1_img(img_tokens)
        lbl_norm1 = self.norm1_lbl(lbl_tokens)

        attn_output_img, attn_output_lbl = self.attention(img_norm1, lbl_norm1)

        img_tokens = img_identity + attn_output_img
        lbl_tokens = lbl_identity + attn_output_lbl

        img_identity = img_tokens
        lbl_identity = lbl_tokens

        tokens = torch.cat((img_tokens, lbl_tokens), dim=1)
        tokens_norm = self.norm2(tokens)

        tokens_mlp = self.mlp(tokens_norm)

        img_tokens = img_identity + tokens_mlp[:, :img_identity.shape[1]]
        lbl_tokens = lbl_identity + tokens_mlp[:, img_identity.shape[1]:]

        return img_tokens, lbl_tokens


# --- LT Vision Transformer ---
class LT_VisionTransformer(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, num_heads, num_classes, num_layers, num_lt_layers, hidden_dim, dropout_rate, in_channels=1):
        """
        Args:
            input_size (int): Length of the input sequence (e.g., time series length or flattened image dim).
            patch_size (int): Size of each patch for 1D Conv embedding.
            embed_dim (int): Dimension of token embeddings.
            num_heads (int): Number of attention heads.
            num_classes (int): Number of output classes (labels).
            num_layers (int): Total number of transformer blocks.
            num_lt_layers (int): Number of LT_TransformerBlocks at the end (N2 in paper [cite: 39]).
            hidden_dim (int): Hidden dimension of the MLP in transformer blocks.
            dropout_rate (float): Dropout rate.
            in_channels (int): Number of input channels (e.g., 1 for grayscale).
        """
        super(LT_VisionTransformer, self).__init__()
        assert num_lt_layers <= num_layers, "num_lt_layers cannot exceed num_layers"
        self.num_classes = num_classes
        self.num_lt_layers = num_lt_layers
        self.num_standard_layers = num_layers - num_lt_layers # N1 = L - N2 [cite: 39]

        num_patches = input_size // patch_size
        self.num_patches = num_patches

        self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.label_tokens = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)

        self.standard_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout_rate)
            for _ in range(self.num_standard_layers)
        ])
        self.lt_blocks = nn.ModuleList([
            LT_TransformerBlock(embed_dim, num_heads, hidden_dim, dropout_rate)
            for _ in range(self.num_lt_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.fcs = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_classes)
        ])
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        B = x.shape[0] 

        x = self.patch_embed(x)
        x = x.transpose(1, 2)

        x = x + self.pos_embed 
        x = self.dropout(x) 

        for block in self.standard_blocks:
            x, _ = block(x)

        lbl_tokens = self.label_tokens.expand(B, -1, -1) 

        img_tokens = x
        for block in self.lt_blocks:
            img_tokens, lbl_tokens = block(img_tokens, lbl_tokens)

        lbl_tokens_norm = self.norm(lbl_tokens) 

        logits = []
        for i in range(self.num_classes):
            token_i = lbl_tokens_norm[:, i, :]
            logit_i = self.fcs[i](token_i)
            logits.append(logit_i)

        x = torch.cat(logits, dim=1)

        x = self.sigmoid(x)

        return x

class VisionTransformerCNN(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, num_heads, num_classes, num_layers, hidden_dim, dropout_rate):
        super(VisionTransformerCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.LazyConv1d(16, 4, stride=4),
            nn.ReLU(),
            nn.LazyConv1d(32, 4, stride=4),
            nn.ReLU(),
            nn.LazyConv1d(32, 1)
        )

        num_patches = 32
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout_rate) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.fc = nn.Linear(embed_dim, num_classes)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.cnn(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x + self.pos_embed)

        for i, block in enumerate(self.transformer_blocks):
            x, _ = block(x)

        x = self.norm(x)
        cls_token_representation = x[:, 0]
        x = self.fc(cls_token_representation)
        x = self.sigmoid(x)

        return x


class XRFClassifier(nn.Module):
    def __init__(self, num_outputs=16, input_size=4096):
        super(XRFClassifier, self).__init__()
        self.num_outputs = num_outputs
        self.input_size = input_size

        # Convolutional layers
        # Step 2: Conv1 + MaxPool + ReLU
        # Input: (B, 1, 4096)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=5, padding=2, bias=False)
        # Output: (B, 24, 4096)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output: (B, 24, 2048)
        self.relu1 = nn.ReLU()

        # Step 3 & 4: Conv2 + ReLU + MaxPool
        # Input: (B, 24, 2048)
        self.conv2 = nn.Conv1d(in_channels=24, out_channels=48, kernel_size=5, padding=2, bias=False)
        # Output: (B, 48, 2048)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output: (B, 48, 1024)

        # Step 5 & 6: Conv3 + ReLU + MaxPool
        # Input: (B, 48, 1024)
        self.conv3 = nn.Conv1d(in_channels=48, out_channels=96, kernel_size=5, padding=2, bias=False)
        # Output: (B, 96, 1024)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output: (B, 96, 512)

        # Step 7: Conv4 (stride=1 to maintain compatibility with Conv5 based on original logic)
        # Input: (B, 96, 512)
        self.conv4 = nn.Conv1d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2, bias=False)
        # Output: (B, 192, 512)

        self.relu4 = nn.ReLU()
        # Step 8: Conv5 (replaces MaxPool and ReLU)
        # Input (B, 192, 512). k=4, s=4 -> floor((512 - 4)/4 + 1) = floor(508/4 + 1) = 127 + 1 = 128.
        self.conv5 = nn.Conv1d(in_channels=192, out_channels=192, kernel_size=4, stride=4, bias=False)
        self.relu5 = nn.ReLU()
        # Output: (B, 192, 128)

        self.conv1d = nn.LazyConv1d(1, 1)

        # Step 9: Flatten + FC1 (Dense layer)
        self.flatten = nn.Flatten()
        # Calculate flattened size based on the output of conv5
        # Output shape of conv5 is (B, 192, 128)
        flattened_size = 128
        # Note: Training of this layer is staged as per description.

        # Step 10: FC2 (Output layer)
        self.fc2 = nn.Linear(in_features=flattened_size, out_features=self.num_outputs, bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size) or (batch_size, 1, input_size).
                               Preprocessing (scaling, noise) should be done before passing here.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs).
        """
        # Ensure input has channel dimension: (batch_size, 1, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Verify input size
        # assert x.shape[2] == self.input_size, f"Input size mismatch: expected {self.input_size}, got {x.shape[2]}"

        # Conv blocks
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)

        # Flatten and Dense layers
        x = self.conv1d(x)
        x = self.flatten(x)
        x = self.fc2(x)

        return self.sigmoid(x)
    
class ResidualBlock1D(nn.Module):
    """
    A residual block for 1D CNNs.

    Uses Conv1D -> ReLU -> Conv1D structure.
    Includes a shortcut connection that adds the input to the output.
    If input and output channels differ, a 1x1 convolution is used
    on the shortcut path to match dimensions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the convolutional layers.
        padding (int): Padding for the convolutional layers.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(ResidualBlock1D, self).__init__()

        # Main path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        # Shortcut connection
        # We need a shortcut connection if the number of channels changes.
        # This 1x1 conv ensures the shortcut tensor has the same number of
        # channels as the main path's output.
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut_conv = None # Use identity if channels match

        # Final ReLU after addition
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after residual connection and activation.
        """
        # Main path
        residual = x # Store the original input for the shortcut
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        # Shortcut path
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(residual)
        else:
            shortcut = residual # Identity shortcut

        # Add shortcut to main path output
        out += shortcut
        # Apply final ReLU
        out = self.relu2(out)
        return out

class ResNet1D(nn.Module):
    """
    A ResNet-like 1D CNN model for processing sequence data like XRF spectra.

    Uses ResidualBlock1D and MaxPool1D layers to extract features, reducing
    the sequence length from 4096 to 256.
    Includes a final head with a 1x1 convolution to reduce channels,
    flattening, and a linear layer for final output (e.g., classification).
    Avoids spatial aggregation like Global Average Pooling before the linear layer.

    Args:
        input_channels (int): Number of channels in the input signal (e.g., 1 for raw spectrum).
        num_outputs (int): Number of output units for the final linear layer.
        head_channels (int): Number of channels after the 1x1 conv head (default: 64).
                             Lowering this reduces parameters in the linear layer.
    """
    def __init__(self, input_channels: int = 1, num_outputs: int = 10, head_channels: int = 8):
        super(ResNet1D, self).__init__()

        self.num_outputs = num_outputs
        self.head_channels = head_channels
        self.final_length = 256 # Length after the last pooling layer

        # --- Feature Extractor Backbone ---
        # Input shape: (batch_size, input_channels, 4096)

        # Initial convolution and pooling layer
        self.initial_conv = nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3)
        self.initial_relu = nn.ReLU()
        self.initial_pool = nn.MaxPool1d(kernel_size=2, stride=2) # 4096 -> 2048
        # Output shape: (B, 64, 2048)

        # Residual Block 1 + Pooling
        self.res_block1 = ResidualBlock1D(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # 2048 -> 1024
        # Output shape: (B, 64, 1024)

        # Residual Block 2 + Pooling
        self.res_block2 = ResidualBlock1D(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) # 1024 -> 512
        # Output shape: (B, 128, 512)

        # Residual Block 3 + Pooling
        self.res_block3 = ResidualBlock1D(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) # 512 -> 256
        # Output shape: (B, 256, 256)

        # --- Head ---
        # Use 1x1 Convolution to reduce channels without spatial aggregation
        self.head_conv = nn.Conv1d(in_channels=256, out_channels=self.head_channels, kernel_size=1, stride=1, padding=0)
        self.head_relu = nn.ReLU()
        # Output shape: (B, head_channels, 256)

        # Flatten the output of the head convolution
        # Flattened size = head_channels * final_length
        self.flatten = nn.Flatten()
        flattened_features = self.head_channels * self.final_length

        # Final Linear Layer
        self.fc = nn.Linear(flattened_features, num_outputs)
        # Output shape: (B, num_outputs)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, 4096).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs).
        """
        # Input validation
        if x.dim() != 3 or x.shape[1] != self.initial_conv.in_channels or x.shape[2] != 4096:
             raise ValueError(
                 f"Expected input shape (batch_size, {self.initial_conv.in_channels}, 4096), "
                 f"but got {x.shape}"
             )

        # --- Feature Extractor ---
        x = self.initial_conv(x)
        x = self.initial_relu(x)
        x = self.initial_pool(x) # (B, 64, 2048)

        x = self.res_block1(x)
        x = self.pool1(x)      # (B, 64, 1024)

        x = self.res_block2(x)
        x = self.pool2(x)      # (B, 128, 512)

        x = self.res_block3(x)
        x = self.pool3(x)      # (B, 256, 256)

        # --- Head ---
        x = self.head_conv(x)  # (B, head_channels, 256)
        x = self.head_relu(x)
        x = self.flatten(x)    # (B, head_channels * 256)
        x = self.fc(x)         # (B, num_outputs)
        x = self.sigmoid(x)

        return x

class XRFNet4096(nn.Module):
    """
    Analogical PyTorch CNN architecture for predicting net counts in XRF spectra
    with a 4096-bin input, designed to limit the flattened size before the
    first dense layer to be no more than 4096.

    Main differences from the original 1024-input network:
    - Handles 4096 input bins.
    - Modified convolutional block structure to achieve a smaller spatial
      dimension after convolutions, such that the flattened size (channels * spatial_dim)
      is <= 4096.
    - The number of neurons in the first dense layer is adjusted to match the
      new flattened size.

    Main similarities:
    - 1D Convolutional layers, ReLU, MaxPool layers.
    - No bias terms in any layer.
    - No Batch Normalization or Dropout layers.
    - Ends with two dense layers predicting 58 output values.
    - Optional Softplus activation on the output layer for post-training application.

    Note: Data scaling, Poisson noise addition, L1 loss scaling, optimizer setup,
    and the two-stage training process are part of the training script.
    """
    def __init__(self, num_outputs=16):
        super(XRFNet4096, self).__init__()

        self.conv1 = nn.LazyConv1d(out_channels=64, kernel_size=7, stride=1, padding=3, bias=False) 
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4) # 4096 -> 1024 spatial

        self.conv2 = nn.LazyConv1d(out_channels=128, kernel_size=5, stride=1, padding=2, bias=False) 
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4) # 2048 -> 256 spatial

        self.conv3 = nn.LazyConv1d(out_channels=192, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4) # 256 -> 64 spatial

        self.conv4 = nn.LazyConv1d(out_channels=192, kernel_size=5, stride=1, padding=2, bias=False) 
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4) # 64 -> 16 spatial


        self.flatten = nn.Flatten()
        self._flattened_features = 192 * 16 


        self.fc2 = nn.Linear(in_features=self._flattened_features, out_features=num_outputs, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, apply_softplus=False):
        """
        Defines the forward pass of the XRFNet4096 model.

        Args:
            x (torch.Tensor): The input spectrum tensor. Expected shape (batch_size, 1, 4096).
                              Scaling and Poisson noise are assumed to be applied beforehand.
            apply_softplus (bool): If True, applies the Softplus activation to the final output.
                                   Intended for post-training phase or inference.

        Returns:
            torch.Tensor: The predicted net counts. Shape (batch_size, 58).
        """
        # Input x shape: (batch_size, 1, 4096)

        x = self.conv1(x)
        x = self.relu1(x) # ReLU applied before pooling
        x = self.pool1(x) # Shape: (batch_size, 24, 2048)

        x = self.conv2(x)
        x = self.relu2(x) # ReLU applied before pooling
        x = self.pool2(x) # Shape: (batch_size, 48, 1024)

        x = self.conv3(x)
        x = self.relu3(x) # ReLU applied before pooling
        x = self.pool3(x) # Shape: (batch_size, 96, 512)

        x = self.conv4(x)
        x = self.relu4(x) # ReLU applied before pooling
        x = self.pool4(x) # Shape: (batch_size, 192, 256)

        # Final strided convolution layer

        x = self.flatten(x) # Shape: (batch_size, 3840)

        x = self.fc2(x) # Shape: (batch_size, 58) - Final output layer

        # Apply Softplus if requested (for post-training/inference)
        x = self.sigmoid(x)

        return x



def select_model(model_name, device, **kwargs):
    if model_name == 'VisionTransformer':
        embed_dim = kwargs.get('embed_dim', 256)
        model = VisionTransformer(
            input_size=kwargs.get('input_size', 4096),
            patch_size=kwargs.get('patch_size', 128),
            embed_dim=embed_dim,
            num_heads=kwargs.get('num_heads', 8),
            num_classes=kwargs.get('num_classes', len(Elements.LINES)),
            num_layers=kwargs.get('num_layers', 6),
            hidden_dim=kwargs.get('hidden_dim', 4 * embed_dim), 
            dropout_rate=kwargs.get('dropout_rate', 0.1)
        )
    elif model_name == 'XRFClassifier':
        model = XRFClassifier(
            num_outputs=kwargs.get('num_outputs', len(Elements.LINES))
        )
    elif model_name == 'VisionTransformerCNN':
         model = VisionTransformerCNN(
            input_size=kwargs.get('input_size', 4096),
            patch_size=kwargs.get('patch_size', 256),
            embed_dim=kwargs.get('embed_dim', 256),
            num_heads=kwargs.get('num_heads', 8),
            num_classes=kwargs.get('num_classes', len(Elements.LINES)),
            num_layers=kwargs.get('num_layers', 6),
            hidden_dim=kwargs.get('hidden_dim', 4 * kwargs.get('embed_dim', 256)),
            dropout_rate=kwargs.get('dropout_rate', 0.1)
        )
    elif model_name == 'ResNet1D':
         model = ResNet1D(
            input_channels=kwargs.get('input_channels', 1),
            num_outputs=kwargs.get("num_outputs", len(Elements.LINES)),
            head_channels=kwargs.get("head_channels", len(Elements.LINES))
        )
    elif model_name == 'XRFNet4096':
         model = XRFNet4096(
            num_outputs=kwargs.get('num_outputs', len(Elements.LINES)),
        )
    elif model_name == 'LT_VisionTransformer':
        embed_dim = kwargs.get('embed_dim', 256)
        model = LT_VisionTransformer(
            input_size=kwargs.get('input_size', 4096),
            patch_size=kwargs.get('patch_size', 128),
            embed_dim=embed_dim,
            num_heads=kwargs.get('num_heads', 8),
            num_classes=kwargs.get('num_classes', len(Elements.LINES)),
            num_layers=kwargs.get('num_layers', 8),
            num_lt_layers=kwargs.get('num_lt_layers', 3), 
            hidden_dim=kwargs.get('hidden_dim', 4 * embed_dim),
            dropout_rate=kwargs.get('dropout_rate', 0.1)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.to(device)
    return model

if __name__ == "__main__":
    from Elements import Elements
    kwargs = {}
    embed_dim = kwargs.get('embed_dim', 256)
    model = VisionTransformerCNN(
            input_size=kwargs.get('input_size', 4096),
            patch_size=kwargs.get('patch_size', 256),
            embed_dim=embed_dim,
            num_heads=kwargs.get('num_heads', 8),
            num_classes=kwargs.get('num_classes', len(Elements.LINES)),
            num_layers=kwargs.get('num_layers', 8),
            hidden_dim=kwargs.get('hidden_dim', 4 * embed_dim),

            dropout_rate=kwargs.get('dropout_rate', 0.1)
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    print(model(torch.tensor([1.0]*4096 * 64).reshape(64, 1, -1).to(device))[0].shape)

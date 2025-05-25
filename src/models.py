import torch
import torch.nn as nn
if __name__ == '__main__':
    from Elements import Elements
else:
    from .Elements import Elements

class TransformerMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate):
        super(TransformerMLP, self).__init__()
        self.feedforward = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
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
    def __init__(self, input_size, patch_size, embed_dim, num_heads, num_outputs, num_layers, hidden_dim, dropout_rate, in_channels=1):
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

        self.fc = nn.Linear(embed_dim, num_outputs)

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

        qkv = self.qkv_proj(tokens).chunk(3, dim=-1)
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
    def __init__(self, input_size, patch_size, embed_dim, num_heads, num_outputs, num_layers, num_lt_layers, hidden_dim, dropout_rate, in_channels=1):
        """
        Args:
            input_size (int): Length of the input sequence (e.g., time series length or flattened image dim).
            patch_size (int): Size of each patch for 1D Conv embedding.
            embed_dim (int): Dimension of token embeddings.
            num_heads (int): Number of attention heads.
            num_outputs (int): Number of output classes (labels).
            num_layers (int): Total number of transformer blocks.
            num_lt_layers (int): Number of LT_TransformerBlocks at the end (N2 in paper [cite: 39]).
            hidden_dim (int): Hidden dimension of the MLP in transformer blocks.
            dropout_rate (float): Dropout rate.
            in_channels (int): Number of input channels (e.g., 1 for grayscale).
        """
        super(LT_VisionTransformer, self).__init__()
        print(num_lt_layers)
        print(num_layers)
        assert num_lt_layers <= num_layers, "num_lt_layers cannot exceed num_layers"
        self.num_outputs = num_outputs
        self.num_lt_layers = num_lt_layers
        self.num_standard_layers = num_layers - num_lt_layers 

        num_patches = input_size // patch_size
        self.num_patches = num_patches

        self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.label_tokens = nn.Parameter(torch.zeros(1, num_outputs, embed_dim))
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
            nn.Linear(embed_dim, 1) for _ in range(num_outputs)
        ])


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
        for i in range(self.num_outputs):
            token_i = lbl_tokens_norm[:, i, :]
            logit_i = self.fcs[i](token_i)
            logits.append(logit_i)

        x = torch.cat(logits, dim=1)

        return x

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class VisionTransformerCNN(nn.Module):
    def __init__(self, num_patches, num_heads, num_outputs, num_layers, dropout_rate, overlapping_conv=False, use_gelu=False):
        super(VisionTransformerCNN, self).__init__()
        assert num_patches <= 512
        if overlapping_conv:
            self.cnn = nn.Sequential(
                nn.LazyConv1d(64, 8, padding=2, stride=4, bias=False),
                nn.GELU() if use_gelu else nn.ReLU(),
                DepthwiseSeparableConv1d(64, num_patches, 8, padding=2, stride=4),
                nn.GELU() if use_gelu else nn.ReLU(),
                nn.LazyConv1d(num_patches, 1, bias=False)
            )
        else:
            self.cnn = nn.Sequential(
                nn.LazyConv1d(64, 8, stride=8, bias=False),
                nn.GELU() if use_gelu else nn.ReLU(),
                DepthwiseSeparableConv1d(64, num_patches, 2, stride=2),
                nn.GELU() if use_gelu else nn.ReLU(),
                nn.LazyConv1d(num_patches, 1, bias=False)
            )

        self.embed_dim = 256

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, self.embed_dim * 4, dropout_rate) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        self.fc = nn.Linear(self.embed_dim, num_outputs)


    def forward(self, x):
        x = self.cnn(x)

        cls_tokens = self.cls_token.expand(x.shape[0], 1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x + self.pos_embed)

        for i, block in enumerate(self.transformer_blocks):
            x, _ = block(x)

        x = self.norm(x)
        cls_token_representation = x[:, 0]
        x = self.fc(cls_token_representation)

        return x


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, in_channels: int = 1, num_outputs: int = 10):
        super(ResNet1D, self).__init__()

        self.num_outputs = num_outputs

        self.initial_conv = nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3)
        self.initial_relu = nn.ReLU()
        self.initial_pool = nn.MaxPool1d(kernel_size=2, stride=2) # 4096 -> 2048

        self.res_block1 = ResidualBlock1D(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # 2048 -> 1024

        self.res_block2 = ResidualBlock1D(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) # 1024 -> 512

        self.res_block3 = ResidualBlock1D(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) # 512 -> 256

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(256, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_outputs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.initial_conv.in_channels or x.shape[2] != 4096:
             raise ValueError(
                 f"Expected input shape (batch_size, {self.initial_conv.in_channels}, 4096), "
                 f"but got {x.shape}"
             )

        x = self.initial_conv(x)
        x = self.initial_relu(x)
        x = self.initial_pool(x) # (B, 64, 2048)

        x = self.res_block1(x)
        x = self.pool1(x)       # (B, 64, 1024)

        x = self.res_block2(x)
        x = self.pool2(x)       # (B, 128, 512)

        x = self.res_block3(x)
        x = self.pool3(x)       # (B, 256, 256)

        x = self.avg_pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)  

        return x


class XRFNetCountModel(nn.Module):
    """
    CNN model for inferring XRF net counts, based on the provided architecture.

    Adheres to constraints: no normalization, no dropout, no biases.
    Designed for 1D spectra of initial length 1024.

    Args:
        num_outputs (int): The number of output values (spectral series counts).
                           Based on the description, this is 58.
    """
    def __init__(self, num_outputs: int = 17):
        super(XRFNetCountModel, self).__init__()

        self.num_outputs = num_outputs

        self.pool666 = nn.AvgPool1d(kernel_size=4, stride=4)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=5, stride=1, padding=2, bias=False)

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=24, out_channels=48, kernel_size=5, stride=1, padding=2, bias=False)

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=96, kernel_size=5, stride=1, padding=2, bias=False)

        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2, bias=False)

        self.conv5 = nn.Conv1d(in_channels=192, out_channels=192, kernel_size=4, stride=4, padding=0, bias=False)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=192 * 32, out_features=192 * 32, bias=False)

        self.fc2 = nn.Linear(in_features=192 * 32, out_features=self.num_outputs, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1 or x.shape[2] != 4096:
             raise ValueError(
                 f"Expected input shape (batch_size, 1, 4096), but got {x.shape}"
             )

        # --- Feature Extraction ---
        x = self.pool666(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x) # Output: (B, 24, 512)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x) # Output: (B, 48, 256)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x) # Output: (B, 96, 128)

        x = self.conv4(x) # Output: (B, 192, 128) - Assuming stride 1 to match stated output length
        x = self.conv5(x) # Output: (B, 192, 32) - Replaces MaxPool and ReLU

        # --- Flatten ---
        x = self.flatten(x) # Output: (B, 6144)

        # --- Fully Connected Layers ---
        x = self.fc1(x) # Output: (B, 6144)
        x = self.fc2(x) 

        return x
    
def create_model(model_name: str, **kwargs):
    """
    Factory to create different model instances based on a name.

    Args:
        model_name (str): One of:
            - 'vit'
            - 'lt_vit'
            - 'vit_cnn'
            - 'resnet1d'
            - 'xrfnet'
        **kwargs: Keyword arguments passed to the model constructor.

    Returns:
        nn.Module: Instantiated PyTorch model.
    """

    defaults = {
        "vit": {
            "input_size": 4096,
            "patch_size": 256,
            "embed_dim": 192,
            "num_heads": 8,
            "num_outputs": 17,
            "num_layers": 6,
            "hidden_dim": 192*4,
            "dropout_rate": 0.1,
            "in_channels": 1,
        },
        "lt_vit": {
            "input_size": 4096,
            "patch_size": 16,
            "embed_dim": 256,
            "num_heads": 8,
            "num_outputs": 17,
            "num_layers": 6,
            "num_lt_layers": 2,
            "hidden_dim": 256,
            "dropout_rate": 0.1,
            "in_channels": 1,
        },
        "cnn_vit": {
            "num_patches": 256,
            "num_heads": 8,
            "num_layers": 6,
            "dropout_rate": 0.1,
            "num_outputs": 17,
            "overlapping_conv": True,
            "use_gelu": False,
        },
        "resnet1d": {
            "num_outputs": 17,
        },
        "xrfnet": {
            "num_outputs": 17,
        }
    }

    # Merge default args with overrides
    args = {**defaults.get(model_name, {}), **kwargs}
    model_name = model_name.lower()
    
    if model_name == 'vit':
        return VisionTransformer(**args)
    
    elif model_name == 'lt_vit':
        return LT_VisionTransformer(**args)

    elif model_name == 'cnn_vit':
        return VisionTransformerCNN(**args)

    elif model_name == 'resnet1d':
        return ResNet1D(**args)

    elif model_name == 'xrfnet':
        return XRFNetCountModel(**args)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

if __name__ == "__main__":
    for model_name in ['vit', 'lt_vit', 'cnn_vit', 'resnet1d', 'xrfnet']:
        model = create_model(model_name, num_outputs=17)
        dummy = torch.tensor([1.0]*4096 * 64).reshape(64, 1, -1)
        from torchviz import make_dot
        make_dot(model(dummy), params=dict(list(model.named_parameters()))).render(model_name, format="png")
        print(model(dummy)[0].shape)



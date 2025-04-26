import torch
import torch.nn as nn
import torch.nn.functional as F

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

class VisionTransformerCNN(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, num_heads, num_classes, num_layers, hidden_dim, dropout_rate):
        super(VisionTransformerCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.LazyConv1d(16, 4, stride=4),
            nn.ReLU(),
            nn.LazyConv1d(16, 4, stride=4),
            nn.ReLU(),
        )
        input_size = input_size // 4
        self.vit = VisionTransformer(input_size, patch_size, embed_dim, num_heads, num_classes, num_layers, hidden_dim, dropout_rate)
    
    def forward(self, X):
        return self.vit(self.cnn(X))


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

        # Step 8: Conv5 (replaces MaxPool and ReLU)
        # Input (B, 192, 512). k=4, s=4 -> floor((512 - 4)/4 + 1) = floor(508/4 + 1) = 127 + 1 = 128.
        self.conv5 = nn.Conv1d(in_channels=192, out_channels=192, kernel_size=4, stride=4, bias=False)
        # Output: (B, 192, 128)

        self.conv1d = nn.LazyConv1d(1, 1)

        # Step 9: Flatten + FC1 (Dense layer)
        self.flatten = nn.Flatten()
        # Calculate flattened size based on the output of conv5
        # Output shape of conv5 is (B, 192, 128)
        flattened_size = 128
        self.fc1 = nn.Linear(in_features=flattened_size, out_features=flattened_size, bias=False)
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
        x = self.conv5(x)

        # Flatten and Dense layers
        x = self.conv1d(x)
        x = self.flatten(x)
        x = self.fc1(x)     # Potentially frozen during early training
        x = self.fc2(x)

        return self.sigmoid(x)

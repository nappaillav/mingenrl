import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
import numpy as np

class ResnetStack(nn.Module):
    """ResNet stack module."""

    def __init__(self, inp_channel, num_features: int, num_blocks: int, max_pooling: bool = True):
        super(ResnetStack, self).__init__()
        self.inp_channel = inp_channel
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling

        self.conv1 = nn.Conv2d(
            in_channels=inp_channel,  # Assuming RGB input
            out_channels=self.num_features,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1)
            ) for _ in range(self.num_blocks)
        ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        conv_out = self.conv1(x)

        if self.max_pooling:
            conv_out = F.max_pool2d(conv_out, kernel_size=3, stride=2, padding=1)

        for block in self.blocks:
            block_input = conv_out
            conv_out = block(conv_out)
            conv_out += block_input

        return conv_out


class MLP(nn.Module):
    def __init__(self, hidden_dims: Sequence[int], activate_final: bool = False, layer_norm: bool = False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if i < len(hidden_dims) - 2 or activate_final:
                self.layers.append(nn.ReLU())
                if layer_norm:
                    self.layers.append(nn.LayerNorm(hidden_dims[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    def __init__(self, inp_channel:int = 3, width: int = 1, stack_sizes: tuple = (16, 32, 32), num_blocks: int = 2,
                 dropout_rate: float = None, mlp_hidden_dims: Sequence[int] = (512,), layer_norm: bool = False
                 ):
        super(ImpalaEncoder, self).__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.mlp_hidden_dims = mlp_hidden_dims
        self.layer_norm = layer_norm
        self.inp_channels = [inp_channel] + list(stack_sizes)
        self.stack_blocks = nn.ModuleList([
            ResnetStack(
                inp_channel= self.inp_channels[i],
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ])

        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        if self.layer_norm:
            self.ln = nn.LayerNorm([32, 8, 8]) # 84X84 --> 32X11X11

        # Assuming the input shape is known, you'd calculate the flattened size here
        # For this example, let's assume it flattens to 1024
        flattened_size = 1024
        mlp_dims = [flattened_size] + list(self.mlp_hidden_dims)
        # self.mlp = MLP(mlp_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mlp1 = nn.Linear(2048, 512)
        self.relu = nn.ReLU() 
        self.mlp2 = nn.Linear(512, 128)

    def forward(self, x, train=True):
        x = x.float() / 255.0

        conv_out = x

        for idx, block in enumerate(self.stack_blocks):
            conv_out = block(conv_out)
            if self.dropout_rate is not None and train:
                conv_out = self.dropout(conv_out)
            # print(conv_out.shape)

        conv_out = F.relu(conv_out)
        if self.layer_norm:
            conv_out = self.ln(conv_out)

        out = conv_out.reshape(conv_out.shape[0], -1)
        # out = self.mlp2(self.relu(self.mlp1(out)))
        out = self.mlp1(out)
        return out

class CNNBlock(nn.Module):
    def __init__(self, inp, out, dropout=0.1, batch_norm=False):
        super(CNNBlock, self).__init__()
        if batch_norm:
            self.model = nn.Sequential(
                    nn.Conv2d(inp, out, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm2d(out),
                    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
                )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
            )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.model(x)
    
class Encoder(nn.Module):
    def __init__(self, inp_channel=3, filters=[32, 32, 32], dropout=0.1, image_size=(3, 64, 64), 
                 batch_norm=False, layer_norm=True, out_emb = 128):
        super(Encoder, self).__init__()
        self.inp_channel = inp_channel 
        self.filters = [inp_channel] + filters
        self.layer_norm = layer_norm
        self.image_size = list(image_size)
        self.stack_blocks = nn.ModuleList([
                        CNNBlock(self.filters[i], self.filters[i+1], batch_norm=batch_norm)
                        for i in range(len(self.filters)-1)
                        ])
        ln_size = self._inp_size()
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_nm = nn.LayerNorm(ln_size)
        self.mlp = nn.Linear(np.prod(list(ln_size)), out_emb)

    def forward(self, x):
        for layer in self.stack_blocks:
            x = layer(x)
        if self.layer_norm:
            x = self.layer_nm(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        return x
    
    def _inp_size(self):
        batch_size = 1
        x = torch.zeros([batch_size] + self.image_size)
        for layer in self.stack_blocks:
            x = layer(x)
        return x.shape[1:]

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # # Define model parameters
    # width = 1
    # stack_sizes = (16, 32, 32)
    # num_blocks = 2
    # dropout_rate = 0.1
    # mlp_hidden_dims = (512,)
    # layer_norm = True
    # inp_channel = 3

    # # Create an instance of ImpalaEncoder
    # model = ImpalaEncoder(
    #     inp_channel=inp_channel,
    #     width=width,
    #     stack_sizes=stack_sizes,
    #     num_blocks=num_blocks,
    #     dropout_rate=dropout_rate,
    #     mlp_hidden_dims=mlp_hidden_dims,
    #     layer_norm=layer_norm
    # )

    # # Print model architecture
    # print(model)
    # print(f"Total Number of Parameter : {sum(p.numel() for p in model.parameters())}")
    # # Create a sample input tensor (batch_size, channels, height, width)
    # # Assuming input image size of 84x84 (common in some RL tasks)
    # batch_size = 4
    # input_tensor = torch.rand(batch_size, 3, 64, 64)

    # # Set model to evaluation mode
    # model.eval()

    # # Forward pass
    # with torch.no_grad():
    #     output = model(input_tensor)

    # # Print output shape
    # print(f"Output shape: {output.shape}")

    # # Test with training mode (to check dropout)
    # model.train()
    # output_train = model(input_tensor, train=True)
    # print(f"Output shape (training mode): {output_train.shape}")
    x = torch.zeros(4, 3, 64,64)
    model = Encoder()
    out = model(x)
    print(out.shape)
    print(f"Total Number of Parameter : {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    main()

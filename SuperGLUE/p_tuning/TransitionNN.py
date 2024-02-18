import torch
import torch.nn as nn

class transition_function_transformer(torch.nn.Module):
    def __init__(self, d_model = 1024, dropout=0.1):
        print("init transtion function")
        super().__init__()
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, dim_feedforward = self.d_model, nhead=2
        )
    
    def forward(self, u_w):
        u_w_input = u_w.permute(1, 0, 2)
        trans_output = self.encoder_layer(u_w_input)
        trans_output = trans_output.permute(1, 0, 2)

        return trans_output
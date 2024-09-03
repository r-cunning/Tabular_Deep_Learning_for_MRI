import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def build_encoder(layer_sizes):
    layers = []
    for size in layer_sizes:
        layers.append(nn.Linear(size[0], size[1]))
        layers.append(nn.ReLU())
        
    layers.pop()

    return nn.Sequential(*layers)

def build_decoder(layer_sizes, final_activation=nn.Sigmoid()):
    layers = []
    for size in layer_sizes:
        layers.append(nn.Linear(size[0], size[1]))
        layers.append(nn.ReLU())
    layers.pop()
    if final_activation is not None:
        layers.append(final_activation)
    else:
        layers.append(nn.Linear(layer_sizes[-1][1], layer_sizes[-1][1]))
    return nn.Sequential(*layers)

def build_regressor(layer_sizes):
    layers = []
    for size in layer_sizes:
        layers.append(nn.Linear(size[0], size[1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layer_sizes[-1][1], 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)



class Autoencoder(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, final_activation=nn.Sigmoid):
        super(Autoencoder, self).__init__()
        self.decoder = build_decoder(decoder_layers, final_activation=final_activation)
        self.encoder = build_encoder(encoder_layers)
        
    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded
    
    def encode(self, x):
        x_encoded = self.encoder(x)
        return x_encoded
    
    
class Regressor(nn.Module):
    def __init__(self, regressor_layers):
        super(Regressor, self).__init__()
        self.regressor = build_regressor(regressor_layers)

    def forward(self, x):
        y_hat = self.regressor(x)
        return y_hat
    
    
    
class CombinedAutoencoderRegressionModel(nn.Module):
    def __init__(self, encoders, encoded_dim_list, regression_layers, output_dim=1, freeze_encoders=False):
        super(CombinedAutoencoderRegressionModel, self).__init__()
        self.encoders = nn.ModuleList(encoders)  # List of encoders

        if freeze_encoders:
            for encoder in self.encoders:
                for param in encoder.parameters():
                    param.requires_grad = False
        
        # Calculate the total number of encoded features
        total_encoded_dim = sum(encoded_dim_list)
        # print("total_encoded_dim: ", total_encoded_dim)
        # Define a regression layer
        self.regression = build_regressor(regression_layers)
    
    def forward(self, x):
        # Apply each encoder and concatenate their outputs
        encoded_outputs = [encoder.encoder(x[i]) for i, encoder in enumerate(self.encoders)]
        combined_output = torch.cat(encoded_outputs, dim=1)
        
        # Pass through the regression layer
        output = self.regression(combined_output)
        return output
    
    
    
    

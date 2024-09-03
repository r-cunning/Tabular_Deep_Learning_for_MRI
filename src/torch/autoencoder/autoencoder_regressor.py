

def build_sequential_layer(layer_sizes):
    layers = []
    for size in layer_sizes:
        layers.append(nn.Linear(size[0], size[1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)



class my_dataset(Dataset):
    def __init__(self, data, data_flag = 'train'):
        self.flag = data_flag
        self.data = data[0]
        self.y = data[1]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = self.data[index]
        y = self.y[index]
        sample = {'data': data, 'y': y}
        return sample
    
    
    
class Autoencoder(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, mode):
        super(Autoencoder, self).__init__()
        
        self.encoder = build_sequential_layer(encoder_layers)
        self.decoder = build_sequential_layer(decoder_layers)

        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, dim_model, nhead, num_layers, dim_feedforward=2048):
        super(TransformerEncoder, self).__init__()
        self.input_linear = nn.Linear(input_size, dim_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_linear = nn.Linear(dim_model, input_size)
        self.attention_weights = None

    def forward(self, src):
        src = self.input_linear(src)
        # хук для извлечения весов внимания
        hooks = []
        def hook(module, input, output):
            self.attention_weights = output[1]  # Второй элемент в output это веса внимания

        for layer in self.transformer_encoder.layers:
            hooks.append(layer.self_attn.register_forward_hook(hook))

        output = self.transformer_encoder(src)

        for hook in hooks:
            hook.remove()

        output = self.output_linear(output)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, input_size, dim_model, nhead, num_layers, dim_feedforward=2048):
        super(TransformerDecoder, self).__init__()
        self.input_linear = nn.Linear(input_size, dim_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(dim_model, input_size)
        self.attention_weights = None

    def forward(self, tgt, memory):
        tgt = self.input_linear(tgt)

        # хук для извлечения весов внимания
        hooks = []
        def hook(module, input, output):
            self.attention_weights = output[1]  # Второй элемент в output это веса внимания

        for layer in self.transformer_decoder.layers:
            hooks.append(layer.self_attn.register_forward_hook(hook))

        output = self.transformer_decoder(tgt, memory)

        for hook in hooks:
            hook.remove()

        output = self.output_linear(output)
        return output

class CNN1DEncoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3):
        super(CNN1DEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_size, num_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(num_channels, num_channels * 2, kernel_size, padding=kernel_size // 2)
        self.fc = nn.Linear(num_channels * 2, input_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Перестановка для соответствия формату CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Возврат к исходному формату
        x = self.fc(x)
        return x

class CNN1DDecoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3):
        super(CNN1DDecoder, self).__init__()
        self.conv1 = nn.Conv1d(input_size, num_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(num_channels, num_channels * 2, kernel_size, padding=kernel_size // 2)
        self.fc = nn.Linear(num_channels * 2, input_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Перестановка для соответствия формату CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Возврат к исходному формату
        x = self.fc(x)
        return x
    
class SimpleEncoder(nn.Module):
    def __init__(self, input_size):
        super(SimpleEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.activ = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        return x
    
class SimpleDecoder(nn.Module):
    def __init__(self, input_size):
        super(SimpleDecoder, self).__init__()
        self.activ = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, input_size)

    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        return x
    
class Autoencoder(nn.Module):
    def __init__(self, input_size, architecture='transformer', **kwargs):
        super(Autoencoder, self).__init__()
        self.architecture = architecture
        if architecture == 'transformer':
            self.encoder = TransformerEncoder(input_size, **kwargs)
            self.decoder = TransformerDecoder(input_size, **kwargs)
        elif architecture == 'cnn':
            self.encoder = CNN1DEncoder(input_size, **kwargs)
            self.decoder = CNN1DDecoder(input_size, **kwargs)
        elif architecture == 'simple':
            self.encoder = SimpleEncoder(input_size)
            self.decoder = SimpleDecoder(input_size)
        elif architecture == 'vae':
            pass
        self.attention_weights = None

    def forward(self, x):
        if self.architecture == 'transformer':
            self.encoder(x)
            self.attention_weights = self.encoder.attention_weights
            x = self.decoder(x)
        else:
            x = self.encoder(x)
            x = self.decoder(x)
        return x

    def get_attention_weights(self):
        return self.attention_weights

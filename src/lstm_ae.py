import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        """
        Инициализирует объект класса Encoder.

        Args:
            input_dim (int): Размерность входных данных.
            out_dim (int): Размерность выходных данных (кодирования).
            h_dims (List[int]): Список размерностей скрытых слоев.
            h_activ (torch.nn.Module or None): Функция активации для скрытых слоев.
            out_activ (torch.nn.Module or None): Функция активации для выходного слоя.
        """
        super(Encoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ
        
    def forward(self, x):
        """
        Производит прямой проход через кодировщик.

        Args:
            x (torch.Tensor): Входные данные.

        Returns:
            torch.Tensor: Результат кодирования.
        """
        x = x.unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n).squeeze()

        return h_n.squeeze()


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ):
        """
        Инициализирует объект класса Decoder.

        Args:
            input_dim (int): Размерность входных данных (кодирования).
            out_dim (int): Размерность выходных данных.
            h_dims (List[int]): Список размерностей скрытых слоев.
            h_activ (torch.nn.Module or None): Функция активации для скрытых слоев.
        """
        super(Decoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [h_dims[-1]]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ = h_activ
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], out_dim), dtype=torch.float), requires_grad=True
        )

    def forward(self, x, seq_len):
        """
        Производит прямой проход через декодировщик.

        Args:
            x (torch.Tensor): Входные данные (кодирования).
            seq_len (int): Длина последовательности.

        Returns:
            torch.Tensor: Результат декодирования.
        """
        x = x.repeat(seq_len, 1).unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)

        return torch.mm(x.squeeze(0), self.dense_matrix)

    
class LSTM_AE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        h_dims=[],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh(),
    ):
        """
        Инициализирует объект класса LSTM_AE.

        Args:
            input_dim (int): Размерность входных данных.
            encoding_dim (int): Размерность кодирования.
            h_dims (List[int]): Список размерностей скрытых слоев.
            h_activ (torch.nn.Module or None): Функция активации для скрытых слоев.
            out_activ (torch.nn.Module or None): Функция активации для выходного слоя.
        """
        super(LSTM_AE, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ, out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1], h_activ)

    def forward(self, x):
        """
        Производит прямой проход через автокодировщик.

        Args:
            x (torch.Tensor): Входные данные.

        Returns:
            torch.Tensor: Реконструированные данные.
        """
        seq_len = x.shape[0]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)

        return x
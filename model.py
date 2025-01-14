import torch
from torch.optim import Adam
from torch_geometric.nn import GAE, GCNConv  # type: ignore


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, mid_channels, cached=True)
        self.conv2 = GCNConv(mid_channels, 2 * out_channels, cached=True)
        self.conv3 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)


def model(
    data, mid_channels, out_channels, L2_rate, learning_rate
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.optim.Optimizer]:
    # this by default sets the decoder to be the inner product
    model = GAE(GCNEncoder(data.num_features, mid_channels, out_channels))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index_f.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=L2_rate)

    return (model, x, train_pos_edge_index, optimizer)

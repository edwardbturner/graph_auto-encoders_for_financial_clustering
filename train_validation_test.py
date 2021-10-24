def train(model, optimizer, pos_edge_index):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, pos_edge_index)
    loss = model.recon_loss(z, pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss) 


def validate(model, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, pos_edge_index)
        loss = model.recon_loss(z, pos_edge_index)
        auc, accuracy = model.test(z, pos_edge_index, neg_edge_index)
    return (float(loss), accuracy)


def test(model, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, pos_edge_index)
        auc, accuracy = model.test(z, pos_edge_index, neg_edge_index)
    return accuracy


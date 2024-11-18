import torch 
from torchvision.utils import make_grid

def get_dataset_by_digit(dataset):
    dic = {i:[] for i in range(10)}
    for img, lab in dataset:
        dic[lab].append(img)
    dic = {i:torch.stack(dic[i]) for i in range(10)}
    return dic

def show_batch(model, batch, conv=None, device="cpu"):

    batch = batch[:10]

    batch_in = batch.to(device)
    if (len(batch_in.shape)!= 2) and (conv==None):
        batch_in = batch_in.reshape(-1, 784)

    with torch.no_grad():
        reconstructed, _, _ = model(batch_in)

    if len(batch.shape)!= 4:
        batch = batch.reshape(-1, 1, 28, 28)

    grid_image_normal = make_grid(batch, nrow=10)
    grid_image_normal = grid_image_normal.permute(1, 2, 0)

    grid_image_reconstruct = make_grid(reconstructed.reshape(-1, 1, 28, 28).detach().cpu(), nrow=10)
    grid_image_reconstruct = grid_image_reconstruct.permute(1, 2, 0)
    

    grid_image = torch.cat([grid_image_normal, grid_image_reconstruct])

    return grid_image
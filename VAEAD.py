import torch 
import torch.nn as nn 
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

from utils import get_dataset_by_digit, show_batch
from model import VAE
from loss import LossVAE

DEVICE="mps"

dataset = MNIST(root="../../../coding/Dataset/", train=True, download=False, transform=ToTensor())
testset = MNIST(root="../../../coding/Dataset/", train=False, download=False, transform=ToTensor())

train_size = 40000
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
trainset, valset = random_split(dataset, [train_size, val_size], generator=generator)

train_dic_dataset = get_dataset_by_digit(trainset)
val_dic_dataset = get_dataset_by_digit(valset)
test_dic_dataset = get_dataset_by_digit(testset)

BATCH_SIZE=128
EPOCHS=50
LEARNING_RATE=3e-4

for anormal in range(10):

    ANORMAL = anormal
    NORMAL_DIGITS = [i for i in range(10)]
    NORMAL_DIGITS.remove(ANORMAL)

    normal_trainset = torch.cat([train_dic_dataset[i] for i in NORMAL_DIGITS])
    normal_val = torch.cat([val_dic_dataset[i] for i in NORMAL_DIGITS])

    trainloader = DataLoader(normal_trainset, batch_size=BATCH_SIZE, shuffle=True)
    
    batch = torch.stack([val_dic_dataset[i][0] for i in range(10)])

    model = VAE(in_dim=784, hidden_dim=[512, 256], latent_dim=2)

    grid_image = show_batch(model, batch)
    plt.imshow(grid_image)
    plt.axis('off')

    if not os.path.isdir(f'figures/Anomaly_{ANORMAL}'):
        os.mkdir(f'figures/Anomaly_{ANORMAL}')

    plt.savefig(f'figures/Anomaly_{ANORMAL}/gen2_before_training.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = LossVAE()

    for epoch in range(EPOCHS):
        epoch_loss=0
        for inputs in tqdm(trainloader):
            inputs = inputs.flatten(start_dim=1)
            reconstructed, mu, logvar = model(inputs)
            loss = criterion(inputs, reconstructed, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        print(f"For epoch {epoch+1}/{EPOCHS} ; loss is {epoch_loss}")

    checkpoints = {'state_dict':model.state_dict()}
    torch.save(checkpoints, f'checkpoints/model2_anomaly_{ANORMAL}.pkl')

    grid_image= show_batch(model, batch)
    plt.imshow(grid_image)
    plt.title(f"Epoch : {epoch}")
    plt.axis('off')
    plt.legend()
    plt.savefig(f'figures/Anomaly_{ANORMAL}/gen2_after_training.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

    test_results_mean = {i:None for i in range(10)}

    with torch.no_grad():
        for i in range(10):
            inputs = test_dic_dataset[i].flatten(start_dim=1)
            reconstructed, _, _ = model(inputs)
            test_score = torch.sum(((inputs - reconstructed)**2), dim=1).mean().item()
            test_results_mean[i]=test_score

    plt.bar(test_results_mean.keys(), test_results_mean.values())
    plt.title('Mean scores for each digit')

    plt.savefig(f'figures/Anomaly_{ANORMAL}/mean2_scores.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Calcul p-values : partie Validation

    inputs_val = normal_val.flatten(start_dim=1)
    with torch.no_grad():
        val_reconstructed, _, _ = model(inputs_val)

    val_scores = -torch.sum(((inputs_val - val_reconstructed)**2), dim=1)
    val_scores_sorted, indices = val_scores.sort()

    final_results = {i:[None, None] for i in range(10)}

    for digit in range(10):

        inputs_test = test_dic_dataset[digit].flatten(start_dim=1)
        with torch.no_grad():
            test_reconstructed, _, _ = model(inputs_test)

        test_scores = -torch.sum(((inputs_test - test_reconstructed)**2), dim=1)

        test_p_values = (1 + torch.sum(test_scores.unsqueeze(1) >= val_scores_sorted, dim=1)) / (len(val_scores_sorted) + 1)

        final_results[digit][0] = test_p_values.tolist()
        final_results[digit][1] = len(inputs_test)

    with open(f"p_values/2_{ANORMAL}.json", "w") as file:
        json.dump(final_results, file)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns

# setting the output directories
output_path = "./outputs/"
output_path_data = "./outputs/data"
output_path_graphics = "./outputs/graphics"
output_path_models = "./outputs/models"
# check if the output is already existed, if not create it
list_paths = [output_path, output_path_data, output_path_graphics, output_path_models]
for outputs in list_paths:
    if os.path.isdir(outputs):
        print(f"{outputs} exists.")
    else:
        print(f"{outputs} does not exist, creating it.")
        os.makedirs(outputs)

# make random seed
np.random.seed(42)
torch.manual_seed(42)

# generate (random) data
data = np.random.randn(2000, 100)
data.shape
# save (random) data
np.save("./outputs/data/raw_data_sim.npy", data)

# Show 5 heatmaps of the data
fig, axes = plt.subplots(5, 1, sharex=True, figsize=(16, 10))
for sample, ax in zip(np.arange(5), axes.flat):
    sns.heatmap(
        data[sample].reshape(-1, 100),
        cbar=False,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title("data sample %s" % str(sample + 1))
fig.savefig("./outputs/graphics/data_examples.png")

# split the data into training and test
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
data_train_torch = torch.from_numpy(data_train).float()
data_test_torch = torch.from_numpy(data_test).float()

# define the tiny autoencoder model
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.layers(x)

# initialize the encoder
encoder = Encoder(input_dim=100, hidden_dim=100, embedding_dim=50)
print(encoder)
# Set up the loss function and optimizer
loss_function = nn.MSELoss()
optimization = optim.SGD(encoder.parameters(), lr=0.001, momentum=0.9)
# Initialize the lists for the losses and representations
losses = []
representations_during_training = []
# training of the encoder
for epoch in range(4000):
    optimization.zero_grad()
    outputs = encoder(data_train_torch)
    loss = loss_function(outputs, data_train_torch[:, :50]) # Why :50?
    loss.backward()
    optimization.step()
    # save the loss
    losses.append(loss.item())

    # Save the representations of every 1000th epoch
    if epoch % 1000 == 0:
        with torch.no_grad():
            representations = encoder(data_train_torch)
            representations_during_training.append(representations.cpu().numpy())
torch.save(encoder, "./outputs/models/encoder.pth")

# show the loss of the encoder
fig, ax = plt.subplots()
sns.lineplot(x=range(len(losses)), y=losses, ax=ax)
sns.despine(offset=10, ax=ax)
plt.title("Loss of Encoder")
plt.xlabel("Epoch number")
plt.ylabel("Training loss")
fig.savefig("./outputs/graphics/loss_training.png")

# show the representations of the encoder via heatmaps
representations_training = representations_during_training[3]
fig, axes = plt.subplots(1, 5, sharex=True, figsize=(10, 2))
for sample, ax in zip(np.arange(5), axes.flat):
    sns.heatmap(
        representations_training[sample].reshape(-1, 5),
        cbar=False,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title("Sample %s" % str(sample + 1))
fig.savefig("./outputs/graphics/data_representations_examples.png")


# ---- I didn't check below (deleated them)----
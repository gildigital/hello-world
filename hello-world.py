import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import codecs
from collections import Counter


# This code defines a simple language model using an embedding layer, an LSTM layer,
# and a linear layer, and trains it using the Adam optimizer and cross-entropy loss.

# The training loop iterates over the training data, computes the forward and backward
# passes to update the model's parameters, and prints the loss every 100 steps.
# Finally, the trained model is saved to disk.

# Of course, this is just a simple example, and you can add additional features or
# modify the model architecture as needed for your specific task.


# Define the TextFileDataset class
class TextFileDataset(Dataset):
    def __init__(self, filepath, encoding='utf-8'):
        self.vocab = None
        print(f'Opening file: {filepath}')
        with codecs.open(filepath, 'r', encoding=encoding) as f:
            self.data = [(line.strip(), 0) for line in f]
            # Build the vocabulary using a Counter
            vocab_counter = Counter([word for line, _ in self.data for word in line.split()])
            # Sort the vocabulary by frequency
            vocab = sorted(vocab_counter, key=vocab_counter.get, reverse=True)
            # Limit the vocabulary to the specified size
            vocab = vocab[:vocab_size]
            # Create a mapping from word to index
            self.vocab = {word: index for index, word in enumerate(vocab)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Look up the word in the vocabulary and get its index
        data = [self.vocab[word] for word in self.data[idx][0].split()]
        # Convert the data to a tensor
        data = torch.tensor(data)
        return data


# Define the model
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x is of shape (batch_size, sequence_length)
        x = self.embedding(x)
        # x is now of shape (batch_size, sequence_length, embedding_dim)
        output, (hidden, cell) = self.lstm(x)
        # output is of shape (batch_size, sequence_length, hidden_dim)
        output = self.linear(output)
        # output is of shape (batch_size, sequence_length, vocab_size)
        return output


# Define the vocab_size and num_epochs variables
vocab_size = 10000
num_epochs = 5

# Instantiate the model
model = LanguageModel(vocab_size=vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2)

# Define the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Create the dataset
dataset = TextFileDataset(filepath='C:/Users/gilfo/OneDrive/Documents/Python/Data/archive/enwik9.txt')


def collate_fn(data):
    # Print the data
    print(f'data = {data}')
    # Sort the data by sequence length
    data.sort(key=lambda x: len(x), reverse=True)
    # Unpack the data into separate lists
    sequences, labels = zip(*data)
    # Print the sequences and labels
    print(f'sequences = {sequences}')
    print(f'labels = {labels}')
    # pad the sequences and stack them into a single tensor
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return sequences, labels


# Create the dataloader with the custom collate function
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Training loop
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(dataloader):
        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(x)
        # output is of shape (batch_size, sequence_length, vocab_size)
        # y is of shape (batch_size, sequence_length)
        loss = loss_fn(output.view(-1, vocab_size), y.view(-1))

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Print the loss every so often
        if i % 100 == 0:
            print(f'Epoch {epoch}, step {i}: loss = {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'language_model.pt')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import Compose, Normalize
import asyncio
import copy
import random
import time
import warnings
import json
from collections import OrderedDict

warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLIENTS = 10
NUM_ROUNDS = 10          
EPOCHS_PER_ROUND = 3      
BATCH_SIZE = 64
LEARNING_RATE = 0.001

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using apple metal (MPS) GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def load_datasets(num_clients: int):
    trf = Compose([transforms.ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    remainder = len(trainset) - sum(lengths)
    for i in range(remainder): lengths[i] += 1
    client_datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    
    trainloaders = [
        DataLoader(
            ds, 
            batch_size=BATCH_SIZE, 
            shuffle=True
        ) for ds in client_datasets
    ]
    
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, testloader

def train(net, trainloader, epochs: int, lr: float):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader)
    accuracy = correct / total
    return loss, accuracy

def aggregate_weights(weights: list[OrderedDict]) -> OrderedDict:
    weights_avg = copy.deepcopy(weights[0])
    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))
    return weights_avg

async def client_update_task(client_id: int, trainloader: DataLoader, global_model: nn.Module):
    local_model = copy.deepcopy(global_model).to(DEVICE)

    if client_id == 0:
        delay = random.uniform(8, 12)
        print(f"  [Client {client_id} (Straggler)] Starting training... ")
        await asyncio.sleep(delay)
    else:
        delay = random.uniform(1.5, 3.5)
        print(f"  [Client {client_id}] Starting training... ")
        await asyncio.sleep(delay)

    await asyncio.to_thread(
        train, local_model, trainloader, EPOCHS_PER_ROUND, LEARNING_RATE
    )
    
    print(f"  [Client {client_id}] Finished training.")
    return local_model.state_dict()

async def main():
    print("Starting synchronous federated learning simulation (FedAvg)...")
    
    simulation_start_time = time.time()
    results_log = []

    global_model = Net().to(DEVICE)
    trainloaders, testloader = load_datasets(NUM_CLIENTS)

    for round_num in range(1, NUM_ROUNDS + 1):
        round_start_time = time.time()
        print(f"\n--- Starting Round {round_num}/{NUM_ROUNDS} ---")

        client_tasks = [
            client_update_task(i, trainloaders[i], global_model)
            for i in range(NUM_CLIENTS)
        ]
        client_weights = await asyncio.gather(*client_tasks)
        
        round_duration = time.time() - round_start_time
        print(f"--- Round {round_num} finished. All clients returned. Duration: {round_duration:.2f}s ---")

        print("[Server] Aggregating all client weights...")
        global_weights = aggregate_weights(client_weights)
        global_model.load_state_dict(global_weights)

        loss, accuracy = test(global_model, testloader)
        elapsed_time = time.time() - simulation_start_time
        
        print(f"[Server] Evaluation after round {round_num}:")
        print(f"         Global Model Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")
        
        results_log.append({
            "round": round_num,
            "wall_clock_time": elapsed_time,
            "accuracy": accuracy
        })

    print("\n Synchronous simulation finished")
    
    with open("sync_results.json", "w") as f:
        json.dump(results_log, f, indent=2)
    print("Synchronous results saved to 'sync_results.json'")

if __name__ == "__main__":
    asyncio.run(main())
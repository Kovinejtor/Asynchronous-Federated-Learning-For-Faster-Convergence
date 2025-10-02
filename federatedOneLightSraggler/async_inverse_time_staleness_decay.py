import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize
import asyncio
import copy
import random
import time
import warnings
import json
import math
from collections import OrderedDict

warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLIENTS = 10
NUM_ROUNDS_PER_CLIENT = 10
EPOCHS_PER_ROUND = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EVAL_EVERY = 10

BASE_ALPHA = 0.6
DECAY_RATE = 0.05

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
    for i in range(len(trainset) - sum(lengths)): lengths[i] += 1
    client_datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    trainloaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, testloader

def train(net, trainloader, epochs, lr):
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
    return loss / len(testloader), correct / total

async def client_task(client_id: int, trainloader: DataLoader, global_model_ref: list, global_version_ref: list, update_queue: asyncio.Queue):
    for round_num in range(1, NUM_ROUNDS_PER_CLIENT + 1):
        current_version = global_version_ref[0]
        local_model = copy.deepcopy(global_model_ref[0]).to(DEVICE)
        
        delay = random.uniform(8, 12) if client_id == 0 else random.uniform(1.5, 3.5)
        await asyncio.sleep(delay)
        
        await asyncio.to_thread(
            train, local_model, trainloader, EPOCHS_PER_ROUND, LEARNING_RATE
        )

        print(f"  [Client {client_id}] Round {round_num} finished (trained on model v{current_version}). Sending update. ")
        await update_queue.put((local_model.state_dict(), current_version))

def calculate_dynamic_alpha(staleness: int) -> float:
    return BASE_ALPHA / (1 + staleness * DECAY_RATE)

async def server_task(global_model_ref: list, global_version_ref: list, testloader: DataLoader, update_queue: asyncio.Queue, total_updates: int, results_log: list, simulation_start_time: float):
    start_time_total = time.time()

    for updates_processed in range(1, total_updates + 1):
        client_weights, client_version = await update_queue.get()
        
        server_version = global_version_ref[0]
        staleness = server_version - client_version
        dynamic_alpha = calculate_dynamic_alpha(staleness)
        
        print(f"\n[Server] Received update #{updates_processed}/{total_updates}. Staleness: {staleness}. Dynamic Alpha: {dynamic_alpha:.3f}")

        global_state_dict = global_model_ref[0].state_dict()
        with torch.no_grad():
            for key in global_state_dict.keys():
                global_state_dict[key] = (1 - dynamic_alpha) * global_state_dict[key] + dynamic_alpha * client_weights[key].to(DEVICE)
        global_model_ref[0].load_state_dict(global_state_dict)
        
        global_version_ref[0] += 1

        if updates_processed % EVAL_EVERY == 0 or updates_processed == total_updates:
            loss, accuracy = test(global_model_ref[0], testloader)
            elapsed_time = time.time() - start_time_total
            print(f"[Server] Evaluation after {updates_processed} updates:")
            print(f"         Global Model Loss: {loss:.4f}, Accuracy: {accuracy:.2%}, Time: {elapsed_time:.2f}s")
            results_log.append({
                "updates": updates_processed, "wall_clock_time": elapsed_time, "accuracy": accuracy
            })
            
        update_queue.task_done()

async def main():
    print("Starting asynchronous FL with inverse-time staleness decay simulation...")
    
    simulation_start_time = time.time()
    results_log = []

    global_model_ref = [Net().to(DEVICE)]
    global_version_ref = [0]

    total_updates = NUM_CLIENTS * NUM_ROUNDS_PER_CLIENT
    trainloaders, testloader = load_datasets(NUM_CLIENTS)
    update_queue = asyncio.Queue()

    server = asyncio.create_task(
        server_task(global_model_ref, global_version_ref, testloader, update_queue, total_updates, results_log, simulation_start_time)
    )

    clients = [
        asyncio.create_task(client_task(i, trainloaders[i], global_model_ref, global_version_ref, update_queue))
        for i in range(NUM_CLIENTS)
    ]

    await asyncio.gather(*clients)
    await update_queue.join()
    await server

    print("\n Asynchronous FL with inverse-time staleness decay simulation finished")
    
    with open("async_inverse_time_staleness_decay_results.json", "w") as f:
        json.dump(results_log, f, indent=2)
    print("Staleness-aware results saved to 'async_inverse_time_staleness_decay_results.json'")

if __name__ == "__main__":
    asyncio.run(main())
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

warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLIENTS = 10
NUM_ROUNDS_PER_CLIENT = 10 
EPOCHS_PER_ROUND = 3     
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EVAL_EVERY = 10 

ALPHA = 0.2   

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
    trf = Compose([
        transforms.RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    remainder = len(trainset) - sum(lengths)
    for i in range(remainder): lengths[i] += 1
    client_datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    trainloaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]
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

async def client_task(client_id: int, trainloader: DataLoader, global_model: nn.Module, update_queue: asyncio.Queue, event_log: list, simulation_start_time: float):
    for round_num in range(1, NUM_ROUNDS_PER_CLIENT + 1):
        print(f"  [Client {client_id}] Starting training round {round_num}/{NUM_ROUNDS_PER_CLIENT}...")
        
        local_model = copy.deepcopy(global_model).to(DEVICE)
        round_start_time = time.time() - simulation_start_time
        
        if client_id == 0:
            delay = random.uniform(50, 60)
            await asyncio.sleep(delay)
        else:
            delay = random.uniform(1.5, 3.5)
            await asyncio.sleep(delay)
        
        train_start_time = time.time() - simulation_start_time
        event_log.append({
            "task": f"Client {client_id}", "state": "Simulated Delay",
            "start": round_start_time, "finish": train_start_time
        })
        
        await asyncio.to_thread(
            train, local_model, trainloader, EPOCHS_PER_ROUND, LEARNING_RATE
        )

        train_end_time = time.time() - simulation_start_time
        event_log.append({
            "task": f"Client {client_id}", "state": "Training",
            "start": train_start_time, "finish": train_end_time
        })

        print(f"  [Client {client_id}] Round {round_num} finished. Sending update to server.")
        await update_queue.put(local_model.state_dict())

async def server_task(global_model: nn.Module, testloader: DataLoader, update_queue: asyncio.Queue, total_updates: int, event_log: list, results_log: list, simulation_start_time: float):
    updates_processed = 0
    start_time_total = time.time()

    while updates_processed < total_updates:
        client_state_dict = await update_queue.get()
        start_agg_time = time.time() - simulation_start_time
        updates_processed += 1
        print(f"\n[Server] Received update #{updates_processed}/{total_updates}. Aggregating weights...")

        global_state_dict = global_model.state_dict()
        with torch.no_grad():
            for key in global_state_dict.keys():
                global_state_dict[key] = (1 - ALPHA) * global_state_dict[key] + ALPHA * client_state_dict[key].to(DEVICE)
        global_model.load_state_dict(global_state_dict)

        end_agg_time = time.time() - simulation_start_time
        event_log.append({"task": "Server", "start": start_agg_time, "finish": end_agg_time})

        if updates_processed % EVAL_EVERY == 0 or updates_processed == total_updates:
            loss, accuracy = test(global_model, testloader)
            elapsed_time = time.time() - start_time_total
            print(f"[Server] Evaluation after {updates_processed} updates:")
            print(f"         Global Model Loss: {loss:.4f}, Accuracy: {accuracy:.2%}, Time: {elapsed_time:.2f}s\n")
            
            results_log.append({
                "updates": updates_processed,
                "wall_clock_time": elapsed_time,
                "accuracy": accuracy
            })
            
        update_queue.task_done()

async def main():
    print("Starting basic asynchronous federated learning simulation ...")
    
    simulation_start_time = time.time()
    event_log, results_log = [], []

    global_model = Net().to(DEVICE)
    total_updates = NUM_CLIENTS * NUM_ROUNDS_PER_CLIENT
    trainloaders, testloader = load_datasets(NUM_CLIENTS)
    update_queue = asyncio.Queue()

    server = asyncio.create_task(
        server_task(global_model, testloader, update_queue, total_updates, event_log, results_log, simulation_start_time)
    )

    clients = [
        asyncio.create_task(client_task(i, trainloaders[i], global_model, update_queue, event_log, simulation_start_time))
        for i in range(NUM_CLIENTS)
    ]

    await asyncio.gather(*clients)
    await update_queue.join()
    await server

    print("Asynchronous simulation finished")
    
    with open("basic_async_results.json", "w") as f:
        json.dump(results_log, f, indent=2)
    print("Asynchronous results saved to 'basic_async_results.json'")

if __name__ == "__main__":
    asyncio.run(main())
import time

import numpy as np
import torch


def train_loop(model, train_loader, num_epochs, optimizer, loss_fn, framework):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    start = None
    average_batch_time_per_epoch = []
    for _ in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx == 1:
                start = time.time()
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        end = time.time()
        average_batch_time_per_epoch.append(
            (end - start) / (len(train_loader) - 1)
        )
    average_time = np.mean(average_batch_time_per_epoch)

    print(f"Time per batch in {framework}: {average_time:.2f}")

import os
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from .evaluation import evaluation


def training(train_loader, test_loader,
             model, criterion, optimizer, scheduler, num_epochs,
             device, disease):

    # Initial evaluation
    average_test_loss, accuracy, precision, recall = evaluation(model, test_loader, criterion, device)
    print(f'Initialization - Test Loss: {average_test_loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')

    # Initialize best model and test loss
    best_model = None
    best_test_loss = 20
    best_epoch = None

    # Initialize variable list
    train_loss_list = []
    test_loss_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []

    # Main training loop
    for epoch in range(num_epochs):
        # Set training mode
        model.train()

        # Initialise running loss
        running_loss = 0.0

        # Iterate over batches
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            # Zero all gradients
            optimizer.zero_grad()

            # Move inputs and targets to current device
            inputs, targets = inputs.to(device), targets.to(device)

            # Reshape target (from (batch_size, 2) to (batch_size,))
            targets = targets.squeeze(1).long()

            # Forward pass and loss calculation
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # if epoch == 0 and batch_idx == 0:
            #     print(inputs)
            #     print(targets)
            #     print(outputs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Calculate average loss for the epoch
        average_train_loss = running_loss / len(train_loader)

        # Evaluate on test set
        average_test_loss, accuracy, precision, recall = evaluation(model, test_loader, criterion, device)

        # Step the scheduler with the average train loss
        scheduler.step(average_train_loss)

        # Print average train and test loss for the epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}] - ', end=" ")
        print(f'Train Loss: {average_train_loss:.4f}, Test Loss: {average_test_loss:.4f},', end=" ")
        print(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')

        # Append to criteria list
        train_loss_list.append(average_train_loss)
        test_loss_list.append(average_test_loss)
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)

        # Save model if it has the best test loss so far
        if average_test_loss < best_test_loss:
            best_model = deepcopy(model.state_dict())
            best_test_loss = average_test_loss
            best_epoch = epoch + 1

    if best_model:
        torch.save(best_model, os.path.join("data", "final", f"final_{disease}.pth"))
    torch.save(model.state_dict(), os.path.join("data", "interim", f"{num_epochs}_epochs_{disease}.pth"))

    plt.clf()
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(test_loss_list, label="Test Loss")

    plt.legend()
    plt.title(f"Train and Test Loss for {disease}")
    plt.xlabel("Epoch")
    plt.ylabel("Losses")

    plt.savefig(os.path.join("data", "final", f"losses_{disease}.png"), dpi=300)

    plt.clf()
    plt.plot(accuracy_list, label="Accuracy")
    plt.plot(recall_list, label="Recall")
    plt.plot(precision_list, label="Precision")

    plt.legend()
    plt.title(f"Other Metrics for {disease}")
    plt.xlabel("Epoch")
    plt.ylabel("Losses")

    plt.savefig(os.path.join("data", "final", f"metrics_{disease}.png"), dpi=300)

    print(f"Best epoch: {best_epoch}")
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.nn.functional import softmax

def evaluation(model, test_loader, criterion, device):
    # Set evaluation mode
    model.eval()

    # Initialise test loss
    test_loss = 0.0

    # Initialise list of targets and predictions
    all_predictions = []
    all_targets = []

    testing = False

    # Disable gradient calculation
    with torch.no_grad():

        # Iterate through data
        for inputs, targets in test_loader:

            # Move data to curent device
            inputs, targets = inputs.to(device), targets.to(device)

            # Reshape targets (from (batch_size, 1) to (batch_size,))
            targets = targets.squeeze(1).long()

            # Forward pass and loss calculation
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Get class predictions
            predictions = softmax(outputs, dim=1).argmax(dim=1)

            if testing:
                testing = False
                print(outputs)
                print(predictions)

            # Add targets and predictions to global list
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    # Calculate metrics
    average_test_loss = test_loss / len(test_loader)

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)

    return average_test_loss, accuracy, precision, recall
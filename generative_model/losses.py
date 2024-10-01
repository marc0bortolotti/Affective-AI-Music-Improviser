import torch
import torch.nn as nn

class CrossEntropyWithPenaltyLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_penalty=1.0):
        """
        Custom loss function combining Cross Entropy loss and a penalty based
        on the number of predictions equal to class 0.

        Args:
        - weight_ce: Weight for the Cross Entropy loss term.
        - weight_penalty: Weight for the penalty term.
        """
        super(CrossEntropyWithPenaltyLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_penalty = weight_penalty
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        Args:
        - outputs: Model predictions (logits), shape (batch_size, num_classes)
        - targets: Ground truth labels, shape (batch_size,)
        
        Returns:
        - Combined loss (cross entropy + penalty)
        """

        # Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(outputs, targets)

        # Get predicted class by taking argmax along the class dimension
        predictions = torch.argmax(outputs, dim=1)

        # Penalty: Number of predictions equal to class 0
        zero_class_predictions = (predictions == 0).float().sum()
        zero_class_targets = (targets == 0).float().sum()
        total_predictions = predictions.size(0)

        # Normalize the penalty term
        penalty = abs((zero_class_predictions - zero_class_targets) / total_predictions)

        # Compute the total loss as a weighted sum
        total_loss = self.weight_ce * ce_loss + self.weight_penalty * penalty

        return total_loss
    

class CrossEntropyLoss():

    def __init__(self, weight=None):
        self.loss = nn.CrossEntropyLoss(weight)

    def __call__(self, outputs, targets):
        return self.loss(outputs, targets)
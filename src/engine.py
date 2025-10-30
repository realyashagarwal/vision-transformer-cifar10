import torch
import random
from tqdm import tqdm

# We import our own custom loss function from our utils file
from .utils import mixup_criterion

def train_epoch(model, dataloader, criterion, optimizer, device, mixup_fn=None, cutmix_fn=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply MixUp or CutMix
        if mixup_fn is not None and random.random() < 0.5:
            inputs, targets_a, targets_b, lam = mixup_fn(inputs, targets)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        elif cutmix_fn is not None:
            inputs, targets_a, targets_b, lam = cutmix_fn(inputs, targets)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            lam = 1
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy (approximate for mixup/cutmix)
        _, predicted = outputs.max(1)
        if lam == 1:
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': running_loss/(batch_idx+1)})
    
    accuracy = 100. * correct / total if total > 0 else 0
    return running_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating', leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return running_loss / len(dataloader), accuracy
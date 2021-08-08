import torch
import torch.nn.functional as F
import numpy as np


def print_decorated(string, char="="):
    boundary = char * 75
    print("\n" + boundary)
    print("%s" % string)
    print(boundary)


def set_torch_seeds(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def eval(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, indexes in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)

    accuracy = correct / len(data_loader.dataset)
    return loss, accuracy, correct


def adjust_learning_rate(epoch, epoch_steps, epoch_decay, optimizer):
    """decrease the learning rate"""

    if epoch in epoch_steps:
        current_lr = optimizer.param_groups[0]["lr"]
        optimizer.param_groups[0]["lr"] = current_lr * epoch_decay
        print(
            "=" * 60
            + "\nChanging learning rate to %g\n" % (current_lr * epoch_decay)
            + "=" * 60
        )


def get_random_labels(orig_targets, num_classes, noise_rate):

        y_targeted = orig_targets

        if noise_rate > 0:

            noise_mask = (
                torch.FloatTensor(size=orig_targets.shape).uniform_(0, 1)
                < noise_rate
            )
            rand_labels = torch.randint(high=num_classes, size=orig_targets.shape)
            rand_labels = torch.fmod(rand_labels + orig_targets, num_classes)
            y_targeted = torch.where(noise_mask, rand_labels, orig_targets)

        return y_targeted


# Rampup Methods
def sigmoid_rampup(current, rampup_length, phase_shift=-5.0, min_val=0, max_val=1.):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return max_val
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return min_val + (max_val - min_val) * float(np.exp(phase_shift * phase * phase))


def log_rampup(current, rampup_length, warmup_length=3, min_val=0.50, max_val=0.7):

    if current <= warmup_length:
        return min_val
    elif current >= rampup_length:
        return max_val
    else:
        return min_val + (max_val - min_val) * (np.log(current - warmup_length) / + np.log(rampup_length - warmup_length))


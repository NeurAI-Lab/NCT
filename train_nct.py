import argparse
import os
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from utilities.losses import cross_entropy, distillation
from utilities import utils
from models.preact_resnet import PreActResNet18
from models.selector import select_model
from data_prep.cifar import CIFAR10, CIFAR100
from data_prep.torchlist import ImageFilelist

parser = argparse.ArgumentParser(description="Noisy Concurrent Training")
# Model options
parser.add_argument("--exp_identifier", type=str, default="")
parser.add_argument("--model1_architecture", type=str, default="PreActResNet18")
parser.add_argument("--model2_architecture", type=str, default="PreActResNet18")
parser.add_argument("--resnet_multiplier", type=int, default=1)
parser.add_argument("--dataset", type=str, default="CIFAR10")
parser.add_argument("--nthread", type=int, default=4)
# Training options
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--epoch_step", nargs="*", type=int, default=[60, 120, 160])
parser.add_argument("--lr_decay_ratio", type=float, default=0.2)
parser.add_argument("--weight_decay", type=float, default=0.0005)
parser.add_argument("--seeds", nargs="*", type=int, default=[0, 10, 20])
# Dynamic Balancing
parser.add_argument("--temperature", default=4, type=float)
parser.add_argument("--alpha", default=0.9, type=float)
parser.add_argument('--rampup_length', default=20, type=int)
parser.add_argument('--start_val', type=float, default=0.0)
parser.add_argument('--phase_shift', type=float, default=-5.0)
# Target variability
parser.add_argument("--random_label_corruption", type=float, default=0)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--rlc_warmup_period", type=int, default=3)
parser.add_argument("--rlc_min", type=float, default=0.2)
parser.add_argument("--rlc_max", type=float, default=0.7)
# Synthetic Noise Simulation
parser.add_argument('--noise_sim', type=str, default='coteaching', choices=['coteaching', 'divmix'])
parser.add_argument('--noise_type', type=str, default='pairflip', choices=['clean', 'pairflip', 'symmetric'])
parser.add_argument('--noise_rate', type=float, default=0.2)
# storage options
parser.add_argument("--enable_save_epoch", default=290, type=int)
parser.add_argument("--save_freq", default=1, type=int)
parser.add_argument("--dataroot", type=str, default="data")
parser.add_argument("--tiny_imagenet_path", type=str, default="/data/input/datasets/tiny_imagenet/tiny-imagenet-200")
parser.add_argument("--output_dir", type=str, default="experiments")
parser.add_argument("--checkpoint", default="", type=str)
# Device options
parser.add_argument("--cuda", action="store_true")
# evaluation options
parser.add_argument("--train_eval_freq", type=int, default=1)
parser.add_argument("--test_eval_freq", type=int, default=1)


# =============================================================================
# Helper Functions
# =============================================================================
def monitor_clean_noisy_performance(args, model, device, data_loader, is_clean):

    model.eval()

    loss_all = 0
    correct_all = 0

    clean_loss = 0
    noisy_loss = 0

    clean_correct = 0
    noisy_correct = 0

    total_clean = is_clean.sum()
    total_noisy = len(is_clean) - total_clean

    with torch.no_grad():
        for data, target, indexes in data_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)

            ce_losses = F.cross_entropy(output, target, reduce=False).cpu().numpy()

            loss_all += ce_losses.sum().item()
            clean_loss += ce_losses[is_clean[indexes]].sum().item()

            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).cpu().numpy()

            correct_all += correct.sum().item()
            clean_correct += correct[is_clean[indexes]].sum().item()

            if total_noisy:
                noisy_loss += ce_losses[~is_clean[indexes]].sum().item()
                noisy_correct += correct[~is_clean[indexes]].sum().item()

    loss_all /= len(data_loader.dataset)
    clean_loss /= total_clean

    acc_all = correct_all / len(data_loader.dataset)
    clean_acc = clean_correct / total_clean

    if total_noisy:
        noisy_loss /= total_noisy
        noisy_acc = noisy_correct / total_noisy

    else:
        noisy_acc = 0

    return loss_all, acc_all, clean_loss, clean_acc, noisy_loss, noisy_acc


def eval_ensemble(args, model1, model2, device, data_loader):

    model1.eval()
    model2.eval()

    correct_m1 = 0
    correct_m2 = 0
    correct_en = 0

    loss_m1 = 0
    loss_m2 = 0
    loss_en = 0

    total = 0

    with torch.no_grad():
        for inputs, targets, indexes in data_loader:

            inputs, targets = inputs.to(device), targets.to(device)

            outputs_m1 = model1(inputs)
            outputs_m2 = model2(inputs)

            loss_m1 += F.cross_entropy(outputs_m1, targets).item()
            _, predicted_m1 = torch.max(outputs_m1, 1)
            correct_m1 += predicted_m1.eq(targets).cpu().sum().item()

            loss_m2 += F.cross_entropy(outputs_m2, targets).item()
            _, predicted_m2 = torch.max(outputs_m2, 1)
            correct_m2 += predicted_m2.eq(targets).cpu().sum().item()

            outputs_en = outputs_m1 + outputs_m2
            loss_en += F.cross_entropy(outputs_en, targets).item()
            _, predicted_en = torch.max(outputs_en, 1)
            correct_en += predicted_en.eq(targets).cpu().sum().item()

            total += targets.size(0)

    loss_m1 /= len(data_loader.dataset)
    loss_m2 /= len(data_loader.dataset)
    loss_en /= len(data_loader.dataset)

    acc_m1 = correct_m1 / total
    acc_m2 = correct_m2 / total
    acc_en = correct_en / total

    return acc_m1, loss_m1, acc_m2, loss_m2, acc_en, loss_en,


def train_mc(
        args,
        model1,
        model2,
        device,
        train_loader,
        optimizer_m1,
        optimizer_m2,
        epoch,
        writer,
):

    model1.train()
    model2.train()

    train_loss_m1 = 0
    correct_m1 = 0

    train_loss_m2 = 0
    correct_m2 = 0

    total = 0

    num_batches = len(train_loader)

    for batch_idx, (data, target, index) in tqdm(
        enumerate(train_loader), desc="batch training", total=num_batches
    ):

        iteration = (epoch * num_batches) + batch_idx

        # target variability
        target1 = utils.get_random_labels(target, args.num_classes, args.random_label_corruption)
        target2 = utils.get_random_labels(target, args.num_classes, args.random_label_corruption)

        data, target, target1, target2 = data.to(device), target.to(device), target1.to(device), target2.to(device)

        optimizer_m1.zero_grad()
        optimizer_m2.zero_grad()

        out_m1 = model1(data)
        out_m2 = model2(data)

        # Dynamic alpha
        alpha = utils.sigmoid_rampup(epoch, args.rampup_length, args.phase_shift) * args.alpha

        # Loss evaluation for Model 1
        l_ce_m1 = cross_entropy(out_m1, target1)
        l_kl_m1 = distillation(out_m1, out_m2.detach(), args.temperature)
        loss_m1 = (1.0 - alpha) * l_ce_m1 + alpha * l_kl_m1

        writer.add_scalar("model1/alpha", alpha, iteration)
        writer.add_scalar("model1/l_ce", l_ce_m1.item(), iteration)
        writer.add_scalar("model1/l_kl", l_kl_m1.item(), iteration)
        writer.add_scalar("model1/loss", loss_m1.item(), iteration)
        writer.add_scalar("model1/rlc", args.random_label_corruption, iteration)

        # Loss evaluation for Model 2
        l_ce_m2 = cross_entropy(out_m2, target2)
        l_kl_m2 = distillation(out_m2, out_m1.detach(), args.temperature)
        loss_m2 = (1.0 - alpha) * l_ce_m2 + alpha * l_kl_m2

        writer.add_scalar("model2/alpha", alpha, iteration)
        writer.add_scalar("model2/l_ce", l_ce_m2.item(), iteration)
        writer.add_scalar("model2/l_kl", l_kl_m2.item(), iteration)
        writer.add_scalar("model2/loss", loss_m2.item(), iteration)
        writer.add_scalar("model2/rlc", args.random_label_corruption, iteration)

        # perform back propagation on Model 1
        loss_m1.backward()
        optimizer_m1.step()

        # perform back propagation on Model 2
        loss_m2.backward()
        optimizer_m2.step()

        train_loss_m1 += loss_m1.data.item()
        train_loss_m2 += loss_m2.data.item()

        _, predicted_m1 = torch.max(out_m1.data, 1)
        correct_m1 += predicted_m1.eq(target.data).cpu().float().sum()

        _, predicted_m2 = torch.max(out_m2.data, 1)
        correct_m2 += predicted_m2.eq(target.data).cpu().float().sum()

        total += target.size(0)

    train_loss_m1 /= num_batches + 1
    acc_m1 = 100.0 * correct_m1 / total

    train_loss_m2 /= num_batches + 1
    acc_m2 = 100.0 * correct_m2 / total

    print("Model 1 Loss: %.3f | Acc: %.3f%% (%d/%d)" % (train_loss_m1, acc_m1, correct_m1, total))
    print("Model 2 Loss: %.3f | Acc: %.3f%% (%d/%d)" % (train_loss_m2, acc_m2, correct_m2, total))


# =============================================================================
# Training Function
# =============================================================================
def solver(args):

    print(args.experiment_name)

    log_dir = os.path.join(args.experiment_name, "logs")
    model_dir = os.path.join(args.experiment_name, "checkpoints")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    test_log = open(os.path.join(args.experiment_name, 'testset_performance.txt'), 'w')
    test_log.write('epoch\tmodel1\tmodel2\tensemble\n')
    test_log.flush()

    log_path = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M"))
    writer = SummaryWriter(log_path)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    print("device: %s" % device)

    if use_cuda:
        torch.cuda.set_device(0)
        cudnn.benchmark = True

    print("==> Preparing data..")
    if args.dataset == 'CIFAR10':

        args.num_classes = 10

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = CIFAR10(root=args.dataroot,
                                download=True,
                                train=True,
                                transform=transform_train,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate,
                                noise_sim=args.noise_sim,
                                )

        test_dataset = CIFAR10(root=args.dataroot,
                               download=True,
                               train=False,
                               transform=transform_test,
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate,
                               )

        is_clean = train_dataset.noise_or_not

    if args.dataset == 'CIFAR100':

        args.num_classes = 100

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        train_dataset = CIFAR100(root=args.dataroot,
                                 download=True,
                                 train=True,
                                 transform=transform_train,
                                 noise_type=args.noise_type,
                                 noise_rate=args.noise_rate,
                                 noise_sim=args.noise_sim,
                                 )

        test_dataset = CIFAR100(root=args.dataroot,
                                download=True,
                                train=False,
                                transform=transform_test,
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate,
                                )

        is_clean = train_dataset.noise_or_not

    if args.dataset == 'imagenet_tiny':

        args.num_classes = 200
        data_root = args.tiny_imagenet_path
        if args.noise_rate > 0:
            train_kv = "train_noisy_%s_%s_kv_list.txt" % (args.noise_type, args.noise_rate)
            is_clean = np.load(os.path.join(data_root, 'noise_or_not_%s_%s.npy' % (args.noise_type, args.noise_rate)))
        else:
            train_kv = "train_kv_list.txt"
            is_clean = np.load(os.path.join(data_root, 'noise_or_not_clean.npy'))

        test_kv = "val_kv_list.txt"

        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                         std=[0.2302, 0.2265, 0.2262])

        train_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, train_kv),
                                      transform=transforms.Compose([transforms.RandomResizedCrop(56),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    normalize,
                                                                    ]))

        test_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, test_kv),
                                     transform=transforms.Compose([transforms.Resize(64),
                                                                   transforms.CenterCrop(56),
                                                                   transforms.ToTensor(),
                                                                   normalize,
                                                                   ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nthread
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nthread
    )

    # deal with student first
    model1 = PreActResNet18(args.num_classes).to(device)
    model2 = PreActResNet18(args.num_classes).to(device)

    optimizer_m1 = SGD(
        model1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    optimizer_m2 = SGD(
        model2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )

    start_epoch = 0
    checkpoint_path = os.path.join(model_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        checkpoint_dict = torch.load(checkpoint_path)
        model1.load_state_dict(checkpoint_dict["model1"])
        model2.load_state_dict(checkpoint_dict["model2"])
        optimizer_m1.load_state_dict(checkpoint_dict["optimizer_m1"])
        optimizer_m2.load_state_dict(checkpoint_dict["optimizer_m2"])
        start_epoch = checkpoint_dict["epoch"]
        print('Checkpoint successfully loaded from epoch %s' % start_epoch)

    running_acc = []
    for epoch in tqdm(range(args.epochs), desc="training epochs"):

        # adjust learning rate for SGD
        utils.adjust_learning_rate(
            epoch, args.epoch_step, args.lr_decay_ratio, optimizer_m1
        )
        utils.adjust_learning_rate(
            epoch, args.epoch_step, args.lr_decay_ratio, optimizer_m2
        )
        writer.add_scalar("model1/lr", optimizer_m1.param_groups[0]["lr"], epoch)
        writer.add_scalar("model2/lr", optimizer_m2.param_groups[0]["lr"], epoch)

        # Dynamic Target Variability
        args.random_label_corruption = utils.log_rampup(
            epoch, args.epochs, args.rlc_warmup_period, args.rlc_min, args.rlc_max
        )

        if epoch < start_epoch:
            continue

        train_mc(args, model1, model2, device, train_loader, optimizer_m1, optimizer_m2, epoch, writer)

        # evaluation
        if epoch % args.train_eval_freq == 0:
            loss_all, acc_all, clean_loss, clean_acc, noisy_loss, noisy_acc =  \
                monitor_clean_noisy_performance(args, model1, args.device, train_loader, is_clean)
            utils.print_decorated(
                "Model 1 | Training: Average loss: {:.4f}, Accuracy: {}% | Clean Loss: {:.4f}, Accuracy: {}% | Noisy Loss: {:.4f}, Accuracy: {}%)".format(
                    loss_all, acc_all * 100, clean_loss, clean_acc * 100, noisy_loss, noisy_acc * 100
                )
            )
            writer.add_scalar("model1/train_loss", loss_all, epoch)
            writer.add_scalar("model1/train_accuracy", acc_all, epoch)
            writer.add_scalar("model1/clean_loss", clean_loss, epoch)
            writer.add_scalar("model1/noisy_loss", noisy_loss, epoch)
            writer.add_scalar("model1/clean_accuracy", clean_acc, epoch)
            writer.add_scalar("model1/noisy_accuracy", noisy_acc, epoch)

            loss_all, acc_all, clean_loss, clean_acc, noisy_loss, noisy_acc = \
                monitor_clean_noisy_performance(args, model2, args.device, train_loader, is_clean)
            utils.print_decorated(
                "Model 2 | Training: Average loss: {:.4f}, Accuracy: {}% | Clean Loss: {:.4f}, Accuracy: {}% | Noisy Loss: {:.4f}, Accuracy: {}%)".format(
                    loss_all, acc_all * 100, clean_loss, clean_acc * 100, noisy_loss, noisy_acc * 100
                )
            )
            writer.add_scalar("model2/train_loss", loss_all, epoch)
            writer.add_scalar("model2/train_accuracy", acc_all, epoch)
            writer.add_scalar("model2/clean_loss", clean_loss, epoch)
            writer.add_scalar("model2/noisy_loss", noisy_loss, epoch)
            writer.add_scalar("model2/clean_accuracy", clean_acc, epoch)
            writer.add_scalar("model2/noisy_accuracy", noisy_acc, epoch)

        if epoch % args.test_eval_freq == 0:

            acc_m1, loss_m1, acc_m2, loss_m2, acc_en, loss_en, = eval_ensemble(args, model1, model2, device, test_loader)

            test_log.write('%s\t%s\t%s\t%s\n' % (epoch, acc_m1, acc_m2, acc_en))
            test_log.flush()

            utils.print_decorated(
                "Model 1 | Test: Average loss: {:.4f}, Accuracy: {}%".format(
                    loss_m1, acc_m1 * 100
                )
            )
            writer.add_scalar("model1/test_loss", loss_m1, epoch)
            writer.add_scalar("model1/test_accuracy", acc_m1, epoch)

            utils.print_decorated(
                "Model 2 | Test: Average loss: {:.4f}, Accuracy: {}%".format(
                    loss_m2, acc_m2 * 100
                )
            )
            writer.add_scalar("model2/test_loss", loss_m2, epoch)
            writer.add_scalar("model2/test_accuracy", acc_m2, epoch)

            utils.print_decorated(
                "Ensemble | Test: Average loss: {:.4f}, Accuracy: {}%".format(
                    loss_en, acc_en * 100
                )
            )
            writer.add_scalar("Ensemble/test_loss", loss_en, epoch)
            writer.add_scalar("Ensemble/test_accuracy", acc_en, epoch)

            if epoch >= args.epochs - 10:
                running_acc.append(acc_en)

                mean_acc = np.array(running_acc).mean()
                writer.add_scalar("Ensemble/running_acc", mean_acc, epoch)

    # get final test accuracy
    _, _, _, _, acc_en, loss_en, = eval_ensemble(args, model1, model2, device, test_loader)
    writer.close()
    test_log.close()

    # save model
    torch.save(model1, os.path.join(model_dir, "final_model1.pt"))
    torch.save(model2, os.path.join(model_dir, "final_model2.pt"))

    return loss_en, acc_en


def main(args):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    prefix = ""
    if args.exp_identifier:
        prefix = "%s_" % args.exp_identifier

    base_name = "%s%s_model1_%s_model2_%s_%sepochs" % (
        prefix,
        args.model1_architecture,
        args.model2_architecture,
        args.dataset,
        args.epochs,
    )

    base_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(base_dir, exist_ok=True)

    # save training arguments
    args_path = os.path.join(base_dir, "args.txt")
    z = vars(args).copy()
    with open(args_path, "w") as f:
        f.write("arguments: " + json.dumps(z) + "\n")

    if len(args.seeds) > 1:

        lst_test_accs = []
        lst_test_loss = []

        for seed in args.seeds:

            print("\n\n----------- SEED {} -----------\n\n".format(seed))
            utils.set_torch_seeds(seed)

            args.experiment_name = os.path.join(
                args.output_dir, base_name, base_name + "_seed" + str(seed)
            )
            txt_path = args.experiment_name + ".txt"

            # check if the seed has been trained
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    next(f)
                    test_accuracy, test_loss = f.readline().strip().split()
                    test_accuracy, test_loss = float(test_accuracy), float(test_loss)

                print(
                    "Seed %s already trained with %s test accuracy and %s test loss"
                    % (seed, test_accuracy, test_loss)
                )
            else:
                test_loss, test_accuracy = solver(args)

            lst_test_accs.append(test_accuracy)
            lst_test_loss.append(test_loss)

            with open(txt_path, "w+") as f:
                f.write("test_acc\ttest_loss\n")
                f.write("%g\t%g\n" % (test_accuracy, test_loss))

        mu = np.mean(lst_test_accs)
        sigma = np.std(lst_test_loss)
        print("\n\nFINAL MEAN TEST ACC: {:02.8f} +/- {:02.8f}".format(mu, sigma))
        file_name = "mean_test_{:02.8f}_pm_{:02.8f}".format(mu, sigma)

        print(len(args.seeds))

        with open(os.path.join(args.output_dir, base_name, file_name), "w+") as f:
            f.write("seed\ttest_acc\ttest_loss\n")
            for i in range(len(args.seeds)):
                f.write(
                    "%d\t%g\t%g\n" % (args.seeds[i], lst_test_accs[i], lst_test_loss[i])
                )

    else:
        utils.set_torch_seeds(args.seeds[0])
        args.experiment_name = os.path.join(
            args.output_dir, base_name, base_name + "_seed" + str(args.seeds[0])
        )

        txt_path = args.experiment_name + ".txt"

        if os.path.exists(txt_path):
            print("Seed %s already trained")
        else:
            test_loss, test_accuracy = solver(args)
            print("\n\nFINAL TEST ACC RATE: {:02.2f}".format(test_accuracy))
            file_name = "final_test_acc_{:02.2f}".format(test_accuracy)
            with open(os.path.join(args.output_dir, base_name, file_name), "w") as f:
                f.write("NA")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

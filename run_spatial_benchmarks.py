import argparse
import copy
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from AdaptiveLocal2DLayerv2 import AdaptiveLocal2DLayer

try:
    from torchvision.ops import DeformConv2d
except Exception:
    DeformConv2d = None


METHODS = ("dense", "alc2d", "cbam", "coordconv", "deform")


def parse_args():
    parser = argparse.ArgumentParser(description="Compact CIFAR-10/LFW benchmark for ALC2D comparisons.")
    parser.add_argument("--dataset", choices=("cifar10", "lfw"), default="cifar10")
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./results_spatial"))
    parser.add_argument("--methods", nargs="+", default=["dense", "alc2d", "cbam", "coordconv", "deform"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0002)
    parser.add_argument("--alc-lr-weights", type=float, default=0.01)
    parser.add_argument("--alc-lr-mu", type=float, default=0.01)
    parser.add_argument("--alc-lr-sigma", type=float, default=0.005)
    parser.add_argument("--dense-hidden", type=int, default=784)
    parser.add_argument("--alc-grid", type=int, default=28)
    parser.add_argument("--alc-sigma", type=float, default=0.12)
    parser.add_argument("--alc-mu-init", choices=("spread", "middle"), default="spread")
    parser.add_argument("--alc-shared-weights", action="store_true")
    parser.add_argument("--alc-head-width", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--scheduler", choices=("onecycle", "fixed"), default="onecycle")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.3)
    parser.add_argument("--onecycle-div-factor", type=float, default=10.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=100.0)
    parser.add_argument("--tune-alc2d", action="store_true")
    parser.add_argument("--tune-sigmas", nargs="+", type=float, default=[0.05, 0.08, 0.1, 0.125])
    parser.add_argument("--tune-lr-weights", nargs="+", type=float, default=[0.01, 0.03])
    parser.add_argument("--tune-lr-mu", nargs="+", type=float, default=[0.01, 0.03])
    parser.add_argument("--tune-lr-sigma", nargs="+", type=float, default=[0.01, 0.03])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def accuracy_from_logits(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).sum().item(), labels.size(0)


def infer_num_classes(dataset):
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    if hasattr(dataset, "class_to_idx"):
        return len(dataset.class_to_idx)
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, list):
            return len(set(int(item) for item in targets))
        return int(torch.as_tensor(targets).max().item()) + 1
    labels = [int(dataset[index][1]) for index in range(len(dataset))]
    return len(set(labels))


def make_grad_scaler(args, device):
    enabled = args.amp and device.type == "cuda"
    amp_module = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_module, "GradScaler", None)
    if grad_scaler_cls is not None:
        return grad_scaler_cls("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def make_dataloaders(args):
    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        image_size = 32 if args.image_size is None else args.image_size
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = datasets.CIFAR10(root=args.data_root, train=True, download=args.download, transform=train_transform)
        test_set = datasets.CIFAR10(root=args.data_root, train=False, download=args.download, transform=test_transform)
    else:
        image_size = 64 if args.image_size is None else args.image_size
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
        train_set = datasets.LFWPeople(
            root=args.data_root,
            split="train",
            image_set="funneled",
            download=args.download,
            transform=train_transform,
        )
        test_set = datasets.LFWPeople(
            root=args.data_root,
            split="test",
            image_set="funneled",
            download=args.download,
            transform=test_transform,
        )

    sample_image, _ = train_set[0]
    input_shape = tuple(sample_image.shape)
    num_classes = infer_num_classes(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, input_shape, num_classes


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AddCoords(nn.Module):
    def forward(self, x):
        batch_size, _, height, width = x.shape
        yy = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype).view(1, 1, height, 1)
        xx = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype).view(1, 1, 1, width)
        yy = yy.expand(batch_size, -1, -1, width)
        xx = xx.expand(batch_size, -1, height, -1)
        return torch.cat([x, xx, yy], dim=1)


class CoordConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_coords = AddCoords()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(self.add_coords(x))


class DeformBNAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if DeformConv2d is None:
            raise RuntimeError("torchvision.ops.DeformConv2d is not available in this environment.")
        self.offset = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        self.deform = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        offsets = self.offset(x)
        x = self.deform(x, offsets)
        x = self.bn(x)
        return self.act(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x):
        weights = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.gate(weights)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        mean_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        attention = torch.cat([mean_map, max_map], dim=1)
        return x * self.gate(self.conv(attention))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


class DenseHead(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, dropout):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class ALCHead(nn.Module):
    def __init__(self, feature_shape, grid_size, num_classes, dropout, sigma_init, mu_init, shared_weights, head_width):
        super().__init__()
        channels, height, width = feature_shape
        self.alc = AdaptiveLocal2DLayer(
            input_size=(channels, height, width),
            output_size=(grid_size, grid_size),
            mu_init=mu_init,
            normed=True,
            layer_norm=False,
            n_embedding=None,
            si_init=(sigma_init, sigma_init),
            shared_weights=shared_weights,
            activ="relu",
        )
        self.dropout = nn.Dropout(dropout)
        hidden_width = head_width if head_width is not None else grid_size * grid_size
        self.classifier = nn.Sequential(
            nn.Linear(grid_size * grid_size, hidden_width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, num_classes),
        )

    def forward(self, x):
        x = self.alc(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


class SpatialBenchmarkNet(nn.Module):
    def __init__(
        self,
        method,
        input_shape,
        num_classes,
        dense_hidden,
        alc_grid,
        alc_sigma,
        dropout,
        alc_mu_init,
        alc_shared_weights,
        alc_head_width,
    ):
        super().__init__()
        channels = input_shape[0]
        stem_cls = CoordConvBNAct if method == "coordconv" else ConvBNAct
        second_cls = DeformBNAct if method == "deform" else ConvBNAct

        self.stem = stem_cls(channels, 32)
        self.block2 = second_cls(32, 32)
        self.pool = nn.MaxPool2d(2)
        self.feature_dropout = nn.Dropout2d(dropout)
        self.cbam = CBAM(32) if method == "cbam" else nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            features = self.extract_features(dummy)
            feature_shape = tuple(features.shape[1:])

        if method == "alc2d":
            self.head = ALCHead(
                feature_shape,
                alc_grid,
                num_classes,
                dropout,
                alc_sigma,
                alc_mu_init,
                alc_shared_weights,
                alc_head_width,
            )
        else:
            in_features = int(np.prod(feature_shape))
            self.head = DenseHead(in_features, dense_hidden, num_classes, dropout)

    def extract_features(self, x):
        x = self.stem(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.feature_dropout(x)
        x = self.cbam(x)
        return x

    def forward(self, x):
        return self.head(self.extract_features(x))


def make_optimizer(model, args):
    mu_params = []
    sigma_params = []
    alc_weight_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "mu_" in name:
            mu_params.append(param)
        elif "sigma_" in name:
            sigma_params.append(param)
        elif "weights" in name:
            alc_weight_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if alc_weight_params:
        param_groups.append({"params": alc_weight_params, "lr": args.alc_lr_weights, "weight_decay": 0.0})
    if mu_params:
        param_groups.append({"params": mu_params, "lr": args.alc_lr_mu, "weight_decay": 0.0})
    if sigma_params:
        param_groups.append({"params": sigma_params, "lr": args.alc_lr_sigma, "weight_decay": 0.0})
    return SGD(param_groups, momentum=args.momentum, nesterov=False)


def make_scheduler(args, optimizer, steps_per_epoch):
    if args.scheduler == "fixed":
        return None
    max_lrs = [group["lr"] for group in optimizer.param_groups]
    return OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.onecycle_pct_start,
        div_factor=args.onecycle_div_factor,
        final_div_factor=args.onecycle_final_div_factor,
        anneal_strategy="cos",
    )


def apply_alc_constraints(model):
    for module in model.modules():
        if hasattr(module, "apply_constraints"):
            module.apply_constraints()


def run_epoch(model, loader, optimizer, scaler, device, amp_enabled, clip_grad, scheduler=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        apply_alc_constraints(model)

        batch_correct, batch_items = accuracy_from_logits(logits.detach(), labels)
        total_loss += loss.item() * batch_items
        total_correct += batch_correct
        total_items += batch_items
    return total_loss / total_items, 100.0 * total_correct / total_items


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    start_time = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_correct, batch_items = accuracy_from_logits(logits, labels)
        total_loss += loss.item() * batch_items
        total_correct += batch_correct
        total_items += batch_items
    elapsed = time.perf_counter() - start_time
    return total_loss / total_items, 100.0 * total_correct / total_items, elapsed


def train_one_run(args, method, seed, train_loader, test_loader, input_shape, num_classes, device):
    set_seed(seed)
    model = SpatialBenchmarkNet(
        method=method,
        input_shape=input_shape,
        num_classes=num_classes,
        dense_hidden=args.dense_hidden,
        alc_grid=args.alc_grid,
        alc_sigma=args.alc_sigma,
        dropout=args.dropout,
        alc_mu_init=args.alc_mu_init,
        alc_shared_weights=args.alc_shared_weights,
        alc_head_width=args.alc_head_width,
    ).to(device)
    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(args, optimizer, len(train_loader))
    scaler = make_grad_scaler(args, device)

    history = []
    best_test_acc = -1.0
    best_epoch = -1
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.amp,
            args.clip_grad,
            scheduler,
        )
        test_loss, test_acc, eval_seconds = evaluate(model, test_loader, device)
        epoch_seconds = time.perf_counter() - epoch_start
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "epoch_seconds": epoch_seconds,
                "eval_seconds": eval_seconds,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"[{args.dataset}] {method} seed={seed} epoch={epoch:03d}/{args.epochs} "
            f"lr={optimizer.param_groups[0]['lr']:.4e} "
            f"train_acc={train_acc:6.2f} test_acc={test_acc:6.2f} best={best_test_acc:6.2f}"
        )

    total_seconds = time.perf_counter() - training_start
    result = {
        "dataset": args.dataset,
        "method": method,
        "seed": seed,
        "epochs": args.epochs,
        "num_classes": num_classes,
        "input_shape": list(input_shape),
        "parameter_count": count_parameters(model),
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "final_test_acc": history[-1]["test_acc"],
        "final_test_loss": history[-1]["test_loss"],
        "train_seconds_total": total_seconds,
        "epoch_seconds_mean": float(np.mean([item["epoch_seconds"] for item in history])),
        "eval_seconds_mean": float(np.mean([item["eval_seconds"] for item in history])),
        "history": history,
        "config": {
            "lr": args.lr,
            "scheduler": args.scheduler,
            "onecycle_pct_start": args.onecycle_pct_start,
            "onecycle_div_factor": args.onecycle_div_factor,
            "onecycle_final_div_factor": args.onecycle_final_div_factor,
            "alc_lr_weights": args.alc_lr_weights,
            "alc_lr_mu": args.alc_lr_mu,
            "alc_lr_sigma": args.alc_lr_sigma,
            "batch_size": args.batch_size,
            "dense_hidden": args.dense_hidden,
            "alc_grid": args.alc_grid,
            "alc_sigma": args.alc_sigma,
            "alc_mu_init": args.alc_mu_init,
            "alc_shared_weights": bool(args.alc_shared_weights),
            "alc_head_width": args.alc_head_width,
            "dropout": args.dropout,
            "clip_grad": args.clip_grad,
        },
    }
    return result


def save_result(result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{result['dataset']}_{result['method']}_seed{result['seed']}"
    if "tag" in result:
        file_name += f"_{result['tag']}"
    file_name += ".json"
    output_path = output_dir / file_name
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {output_path}")


def save_tuning_summary(rows, output_dir, dataset):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": dataset,
        "rows": rows,
    }
    json_path = output_dir / f"{dataset}_alc2d_tuning_summary.json"
    md_path = output_dir / f"{dataset}_alc2d_tuning_summary.md"
    json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        f"# ALC2D Tuning Summary ({dataset})",
        "",
        "| rank | sigma | lr_weights | lr_mu | lr_sigma | mean best acc | std | mean final acc | mean sec | tag |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {row['alc_sigma']:.4f} | {row['alc_lr_weights']:.4f} | {row['alc_lr_mu']:.4f} | "
            f"{row['alc_lr_sigma']:.4f} | {row['best_acc_mean']:.2f} | {row['best_acc_std']:.2f} | "
            f"{row['final_acc_mean']:.2f} | {row['train_seconds_mean']:.1f} | {row['tag']} |"
        )
    md_path.write_text("\n".join(lines))
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


def run_alc2d_tuning(args, train_loader, test_loader, input_shape, num_classes, device):
    rows = []
    combinations = list(
        itertools.product(args.tune_sigmas, args.tune_lr_weights, args.tune_lr_mu, args.tune_lr_sigma)
    )
    for combo_index, (sigma, lr_weights, lr_mu, lr_sigma) in enumerate(combinations, start=1):
        tune_args = copy.deepcopy(args)
        tune_args.methods = ["alc2d"]
        tune_args.alc_sigma = sigma
        tune_args.alc_lr_weights = lr_weights
        tune_args.alc_lr_mu = lr_mu
        tune_args.alc_lr_sigma = lr_sigma
        tag = f"s{sigma:g}_w{lr_weights:g}_mu{lr_mu:g}_si{lr_sigma:g}".replace(".", "p")
        combo_results = []
        print(
            f"[tune {combo_index}/{len(combinations)}] sigma={sigma} lr_weights={lr_weights} lr_mu={lr_mu} lr_sigma={lr_sigma}"
        )
        for seed in tune_args.seeds:
            result = train_one_run(
                tune_args,
                "alc2d",
                seed,
                train_loader,
                test_loader,
                input_shape,
                num_classes,
                device,
            )
            result["tag"] = tag
            save_result(result, tune_args.output_dir)
            combo_results.append(result)

        best_accs = [item["best_test_acc"] for item in combo_results]
        final_accs = [item["final_test_acc"] for item in combo_results]
        train_times = [item["train_seconds_total"] for item in combo_results]
        rows.append(
            {
                "tag": tag,
                "alc_sigma": sigma,
                "alc_lr_weights": lr_weights,
                "alc_lr_mu": lr_mu,
                "alc_lr_sigma": lr_sigma,
                "best_acc_mean": float(np.mean(best_accs)),
                "best_acc_std": float(np.std(best_accs, ddof=1)) if len(best_accs) > 1 else 0.0,
                "final_acc_mean": float(np.mean(final_accs)),
                "train_seconds_mean": float(np.mean(train_times)),
            }
        )

    rows.sort(key=lambda item: item["best_acc_mean"], reverse=True)
    save_tuning_summary(rows, args.output_dir, args.dataset)


def main():
    args = parse_args()
    if args.tune_alc2d:
        args.methods = ["alc2d"]
    invalid_methods = [method for method in args.methods if method not in METHODS]
    if invalid_methods:
        raise ValueError(f"Unknown methods: {invalid_methods}. Valid methods: {METHODS}")
    if "deform" in args.methods and DeformConv2d is None:
        raise RuntimeError("Requested method 'deform' but torchvision.ops.DeformConv2d is unavailable.")

    device = resolve_device(args.device)
    train_loader, test_loader, input_shape, num_classes = make_dataloaders(args)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}, input_shape={input_shape}, num_classes={num_classes}")

    if args.tune_alc2d:
        run_alc2d_tuning(args, train_loader, test_loader, input_shape, num_classes, device)
        return

    for method in args.methods:
        for seed in args.seeds:
            result = train_one_run(args, method, seed, train_loader, test_loader, input_shape, num_classes, device)
            save_result(result, args.output_dir)


if __name__ == "__main__":
    main()
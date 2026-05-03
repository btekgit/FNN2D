import argparse
import csv
import importlib
import time
from typing import Dict, List, Optional, Tuple

import torch

from run_spatial_benchmarks import METHODS, SpatialBenchmarkNet

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None

try:
    thop_profile = importlib.import_module("thop").profile
except Exception:
    thop_profile = None


def parse_methods(raw_methods: str) -> List[str]:
    items = [item.strip().lower() for item in raw_methods.split(",") if item.strip()]
    if not items:
        raise ValueError("No methods provided.")
    if "all" in items:
        return list(METHODS)
    unknown = [item for item in items if item not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid: {METHODS} or 'all'.")
    return items


def resolve_device(cpu_only: bool) -> torch.device:
    if cpu_only:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_input_shape(dataset: str, image_size: int) -> Tuple[int, int, int]:
    if dataset == "cifar10":
        side = image_size if image_size is not None else 32
        return 3, side, side
    side = image_size if image_size is not None else 64
    return 3, side, side


def build_model(method: str, args: argparse.Namespace, input_shape: Tuple[int, int, int]) -> torch.nn.Module:
    return SpatialBenchmarkNet(
        method=method,
        input_shape=input_shape,
        num_classes=args.num_classes,
        dense_hidden=args.dense_hidden,
        alc_grid=args.alc_grid,
        alc_sigma=args.alc_sigma,
        dropout=args.dropout,
        alc_mu_init=args.alc_mu_init,
        alc_shared_weights=False,
        alc_head_width=args.alc_head_width,
    )


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    all_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return all_params, trainable_params


def compute_flops_fvcore(model: torch.nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
    if FlopCountAnalysis is None:
        return {}
    model.eval()
    with torch.no_grad():
        analysis = FlopCountAnalysis(model, sample_input)
        total_flops = float(analysis.total())
    return {"flops": total_flops}


def compute_flops_thop(model: torch.nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
    if thop_profile is None:
        return {}
    model.eval()
    with torch.no_grad():
        macs, params = thop_profile(model, inputs=(sample_input,), verbose=False)
    return {"macs": float(macs), "thop_params": float(params), "flops": float(macs) * 2.0}


def compute_costs(model: torch.nn.Module, sample_input: torch.Tensor, backend: str) -> Dict[str, float]:
    if backend == "none":
        return {}
    if backend == "fvcore":
        return compute_flops_fvcore(model, sample_input)
    if backend == "thop":
        return compute_flops_thop(model, sample_input)

    result = compute_flops_fvcore(model, sample_input)
    if result:
        return result
    return compute_flops_thop(model, sample_input)


def measure_latency(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int],
    batch_size: int,
    device: torch.device,
    warmup: int,
    runs: int,
) -> float:
    model.eval()
    x = torch.randn(batch_size, *input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                model(x)
            torch.cuda.synchronize()
            total_ms = (time.perf_counter() - t0) * 1000.0
        else:
            t0 = time.perf_counter()
            for _ in range(runs):
                model(x)
            total_ms = (time.perf_counter() - t0) * 1000.0

    return total_ms / runs


def parse_timing_batches(raw: str) -> List[int]:
    out = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not out:
        raise ValueError("At least one timing batch size is required.")
    return out


def format_cost_line(costs: Dict[str, float]) -> str:
    if not costs:
        return "FLOPs/MACs: not computed"

    parts = []
    if "macs" in costs:
        parts.append(f"MACs: {costs['macs'] / 1e9:.4f}G")
    if "flops" in costs:
        parts.append(f"FLOPs: {costs['flops'] / 1e9:.4f}G")
    return ", ".join(parts)


def fmt_num(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def print_markdown_table(rows: List[Dict[str, float]], timing_batches: List[int]) -> None:
    headers = ["method", "params_m", "flops_g"]
    for bs in timing_batches:
        headers.append(f"latency_bs{bs}_ms")
        headers.append(f"per_sample_bs{bs}_ms")

    print("\n# Cost Summary Table")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        line = [
            row["method"],
            fmt_num(row.get("params_m"), 3),
            fmt_num(row.get("flops_g"), 4),
        ]
        for bs in timing_batches:
            line.append(fmt_num(row.get(f"latency_bs{bs}_ms"), 3))
            line.append(fmt_num(row.get(f"per_sample_bs{bs}_ms"), 4))
        print("| " + " | ".join(line) + " |")


def save_csv_table(rows: List[Dict[str, float]], timing_batches: List[int], path: str) -> None:
    headers = ["method", "params", "trainable_params", "params_m", "flops", "flops_g", "macs", "macs_g"]
    for bs in timing_batches:
        headers.append(f"latency_bs{bs}_ms")
        headers.append(f"per_sample_bs{bs}_ms")

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def save_markdown_table(rows: List[Dict[str, float]], timing_batches: List[int], path: str) -> None:
    headers = ["method", "params_m", "flops_g"]
    for bs in timing_batches:
        headers.append(f"latency_bs{bs}_ms")
        headers.append(f"per_sample_bs{bs}_ms")

    lines = [
        "# Cost Summary Table",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        line = [
            row["method"],
            fmt_num(row.get("params_m"), 3),
            fmt_num(row.get("flops_g"), 4),
        ]
        for bs in timing_batches:
            line.append(fmt_num(row.get(f"latency_bs{bs}_ms"), 3))
            line.append(fmt_num(row.get(f"per_sample_bs{bs}_ms"), 4))
        lines.append("| " + " | ".join(line) + " |")

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute cost and latency for shallow spatial models.")
    parser.add_argument("--methods", type=str, default="dense,alc2d,cbam,coordconv,deform")
    parser.add_argument("--dataset", choices=("cifar10", "lfw"), default="cifar10")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used for FLOPs sample input")
    parser.add_argument("--timing-batch-sizes", type=str, default="1,32,256,1024")
    parser.add_argument("--timing-runs", type=int, default=50)
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--flops-backend", choices=("auto", "fvcore", "thop", "none"), default="auto")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional path to save CSV summary table")
    parser.add_argument("--save-md", type=str, default=None, help="Optional path to save Markdown summary table")

    parser.add_argument("--dense-hidden", type=int, default=784)
    parser.add_argument("--alc-grid", type=int, default=28)
    parser.add_argument("--alc-sigma", type=float, default=0.12)
    parser.add_argument("--alc-mu-init", choices=("spread", "middle"), default="spread")
    parser.add_argument("--alc-shared-weights", action="store_true")
    parser.add_argument("--alc-head-width", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.3)

    args = parser.parse_args()

    methods = parse_methods(args.methods)
    timing_batches = parse_timing_batches(args.timing_batch_sizes)
    device = resolve_device(args.cpu)
    input_shape = infer_input_shape(args.dataset, args.image_size)

    print("=" * 100)
    print(f"Device: {device}")
    print(f"Dataset profile: {args.dataset}")
    print(f"Input shape (C,H,W): {input_shape}")
    print(f"Methods: {methods}")
    print("=" * 100)

    rows = []

    for method in methods:
        print(f"\nModel: {method}")
        model = build_model(method, args, input_shape).to(device)
        all_params, trainable_params = count_params(model)

        sample_input = torch.randn(args.batch_size, *input_shape, device=device)
        costs = compute_costs(model, sample_input, backend=args.flops_backend)

        print(f"Params (all): {all_params:,}")
        print(f"Params (trainable): {trainable_params:,}")
        print(format_cost_line(costs))

        print(f"Latency avg over {args.timing_runs} runs (warmup {args.timing_warmup}):")
        print(f"  {'Batch':>6}  {'Avg (ms)':>10}  {'Per-sample (ms)':>16}")
        row = {
            "method": method,
            "params": all_params,
            "trainable_params": trainable_params,
            "params_m": all_params / 1e6,
            "flops": costs.get("flops"),
            "flops_g": costs.get("flops", 0.0) / 1e9 if "flops" in costs else None,
            "macs": costs.get("macs"),
            "macs_g": costs.get("macs", 0.0) / 1e9 if "macs" in costs else None,
        }
        for bs in timing_batches:
            try:
                avg_ms = measure_latency(
                    model=model,
                    input_shape=input_shape,
                    batch_size=bs,
                    device=device,
                    warmup=args.timing_warmup,
                    runs=args.timing_runs,
                )
                print(f"  {bs:>6}  {avg_ms:>10.2f}  {avg_ms / bs:>16.4f}")
                row[f"latency_bs{bs}_ms"] = avg_ms
                row[f"per_sample_bs{bs}_ms"] = avg_ms / bs
            except RuntimeError as exc:
                print(f"  {bs:>6}  {'OOM/error':>10}  ({exc})")
                row[f"latency_bs{bs}_ms"] = None
                row[f"per_sample_bs{bs}_ms"] = None
        rows.append(row)

    rows.sort(
        key=lambda item: (
            item.get(f"per_sample_bs{timing_batches[-1]}_ms") is None,
            item.get(f"per_sample_bs{timing_batches[-1]}_ms") or float("inf"),
        )
    )
    print_markdown_table(rows, timing_batches)

    if args.save_csv:
        save_csv_table(rows, timing_batches, args.save_csv)
        print(f"Saved CSV table: {args.save_csv}")
    if args.save_md:
        save_markdown_table(rows, timing_batches, args.save_md)
        print(f"Saved Markdown table: {args.save_md}")


if __name__ == "__main__":
    main()

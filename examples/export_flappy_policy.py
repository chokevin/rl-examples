"""Export a trained Flappy PPO checkpoint to a small C-readable policy file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pufferlib4 import load_checkpoint


def resolve_checkpoint(checkpoint: Path | None, data_dir: Path) -> Path:
    if checkpoint is not None and str(checkpoint) != "latest":
        return checkpoint

    checkpoints = list(data_dir.glob("*.pt")) + list(data_dir.rglob("*.bin"))
    native_dir = Path("experiments") / "native_checkpoints" / "flappy_bird"
    if data_dir != native_dir and native_dir.exists():
        checkpoints.extend(native_dir.rglob("*.bin"))
    checkpoints = sorted(set(checkpoints), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No .pt/.bin checkpoints found in {data_dir}")
    return checkpoints[-1]


def write_array(file, label: str, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().float().reshape(-1).tolist()
    file.write(label)
    for value in values:
        file.write(f" {value:.9g}")
    file.write("\n")


def network_layers(state_dict: dict[str, torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    layer_indices: list[int] = []
    prefix = "network.net."
    for key in state_dict:
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        index_text = key[len(prefix) : -len(".weight")]
        if index_text.isdigit():
            layer_indices.append(int(index_text))

    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index in sorted(layer_indices):
        weight_key = f"network.net.{index}.weight"
        bias_key = f"network.net.{index}.bias"
        if bias_key in state_dict:
            layers.append((state_dict[weight_key], state_dict[bias_key]))
    return layers


def export_policy(
    checkpoint: Path,
    output: Path,
    hidden_size: int | None,
    pipe_gap: float,
    device: str,
) -> None:
    """Convert the PyTorch MLP into text weights the raylib C app can run.

    Local PPO checkpoints use: 4 observations -> linear encoder -> GELU -> action
    logits. Native PufferLib checkpoints use: encoder -> one or more MLP layers ->
    action logits. The C viewer reimplements both formats, so the raylib window
    does not need Python, PyTorch, or PufferLib while it is running.
    """
    loaded = load_checkpoint(checkpoint, device)
    state_dict = loaded.state_dict

    encoder_weight = state_dict["encoder.encoder.weight"]
    encoder_bias = state_dict["encoder.encoder.bias"]
    decoder_weight = state_dict["decoder.decoder.weight"]
    decoder_bias = state_dict["decoder.decoder.bias"]
    hidden_size = hidden_size or int(encoder_weight.shape[0])
    layers = network_layers(state_dict)
    legacy_local_policy = len(layers) == 0 and bool(loaded.metadata)

    if encoder_weight.shape != (hidden_size, 4):
        raise ValueError(f"Unexpected encoder shape: {tuple(encoder_weight.shape)}")
    if encoder_bias.shape != (hidden_size,):
        raise ValueError(f"Unexpected encoder bias shape: {tuple(encoder_bias.shape)}")
    if decoder_weight.shape != (2, hidden_size):
        raise ValueError(f"Unexpected decoder shape: {tuple(decoder_weight.shape)}")
    if decoder_bias.shape != (2,):
        raise ValueError(f"Unexpected decoder bias shape: {tuple(decoder_bias.shape)}")
    for weight, bias in layers:
        if weight.shape != (hidden_size, hidden_size):
            raise ValueError(f"Unexpected network layer shape: {tuple(weight.shape)}")
        if bias.shape != (hidden_size,):
            raise ValueError(f"Unexpected network layer bias shape: {tuple(bias.shape)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        file.write("flappy_policy_v1\n" if legacy_local_policy else "flappy_policy_v2\n")
        file.write(f"hidden_size {hidden_size}\n")
        if not legacy_local_policy:
            file.write(f"network_layers {len(layers)}\n")
        file.write(f"pipe_gap {pipe_gap:.9g}\n")
        write_array(file, "encoder_weight", encoder_weight)
        write_array(file, "encoder_bias", encoder_bias)
        for layer_idx, (weight, bias) in enumerate(layers):
            write_array(file, f"network_{layer_idx}_weight", weight)
            write_array(file, f"network_{layer_idx}_bias", bias)
        write_array(file, "decoder_weight", decoder_weight)
        write_array(file, "decoder_bias", decoder_bias)

    print(f"Exported raylib policy: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Flappy Bird PPO checkpoint for the C/raylib viewer."
    )
    parser.add_argument("checkpoint", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("experiments") / "flappy_bird",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments") / "flappy_bird" / "policy.txt",
    )
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--pipe-gap", type=float, default=220.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_policy(
        checkpoint=resolve_checkpoint(args.checkpoint, args.data_dir),
        output=args.output,
        hidden_size=args.hidden_size,
        pipe_gap=args.pipe_gap,
        device=args.device,
    )


if __name__ == "__main__":
    main()

"""
Export PyTorch LSTM experts to ONNX for MQL5 native inference.
================================================================
Converts .pth files from top_50_experts/ into .onnx files that
MQL5 can load directly with OnnxCreate() / OnnxRun().

Usage:
    python export_expert_onnx.py                          # Export all
    python export_expert_onnx.py --symbol BTCUSD          # Export BTCUSD only
    python export_expert_onnx.py --file expert_rank11_XAUUSD.pth  # Single file
    python export_expert_onnx.py --output-dir MQL5/Files  # Custom output dir
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ============================================================================
# MODEL DEFINITION (must match training exactly)
# ============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,
                            num_layers=2, dropout=0.2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# ============================================================================
# ONNX WRAPPER (eval mode, no dropout, softmax output)
# ============================================================================
class LSTMExportWrapper(nn.Module):
    """Wraps the LSTM for clean ONNX export:
    - Forces eval mode (dropout disabled)
    - Adds softmax so MQL5 gets probabilities directly
    """
    def __init__(self, model):
        super(LSTMExportWrapper, self).__init__()
        self.lstm = model.lstm
        self.fc = model.fc

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]       # last timestep
        out = self.fc(out)        # logits [batch, 3]
        out = torch.softmax(out, dim=1)  # probabilities
        return out


# ============================================================================
# EXPORT
# ============================================================================
def export_single(pth_path: Path, output_dir: Path, input_size=8,
                  hidden_size=128, seq_length=30):
    """Export one .pth to .onnx"""
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size)

    state = torch.load(pth_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    wrapper = LSTMExportWrapper(model)
    wrapper.eval()

    # Dummy input: [batch=1, seq_length=30, features=8]
    dummy = torch.randn(1, seq_length, input_size)

    onnx_name = pth_path.stem + ".onnx"
    onnx_path = output_dir / onnx_name

    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["market_data"],
        output_names=["probabilities"],
        dynamic_axes={
            "market_data": {0: "batch"},
            "probabilities": {0: "batch"}
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Verify
    size_kb = onnx_path.stat().st_size / 1024
    print(f"  OK: {onnx_name} ({size_kb:.0f} KB)")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export experts to ONNX")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Only export experts for this symbol")
    parser.add_argument("--file", type=str, default=None,
                        help="Export a single .pth file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for .onnx files")
    parser.add_argument("--seq-length", type=int, default=30,
                        help="Sequence length (default: 30)")
    args = parser.parse_args()

    base = Path(__file__).parent
    experts_dir = base / "top_50_experts"
    manifest_path = experts_dir / "top_50_manifest.json"

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = base / "onnx_experts"
    out_dir.mkdir(exist_ok=True)

    if args.file:
        # Single file export
        pth = experts_dir / args.file
        if not pth.exists():
            pth = Path(args.file)
        if not pth.exists():
            print(f"ERROR: {args.file} not found")
            sys.exit(1)
        print(f"Exporting {pth.name}...")
        export_single(pth, out_dir, seq_length=args.seq_length)
        return

    # Batch export from manifest
    if not manifest_path.exists():
        print("ERROR: top_50_manifest.json not found")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    experts = manifest["experts"]
    if args.symbol:
        experts = [e for e in experts if e["symbol"] == args.symbol.upper()]
        print(f"Filtering for {args.symbol.upper()}: {len(experts)} experts")

    print(f"Exporting {len(experts)} experts to ONNX...")
    print(f"Output: {out_dir}")
    print(f"Seq length: {args.seq_length}, Input: 8, Hidden: 128, Output: 3")
    print("-" * 60)

    exported = 0
    failed = 0
    for expert in experts:
        pth = experts_dir / expert["filename"]
        if not pth.exists():
            print(f"  SKIP: {expert['filename']} not found")
            failed += 1
            continue

        try:
            export_single(
                pth, out_dir,
                input_size=expert.get("input_size", 8),
                hidden_size=expert.get("hidden_size", 128),
                seq_length=args.seq_length,
            )
            exported += 1
        except Exception as e:
            print(f"  FAIL: {expert['filename']}: {e}")
            failed += 1

    # Also export mutants if they exist
    for mutant in experts_dir.glob("*_MUTANT.pth"):
        try:
            print(f"  [MUTANT] {mutant.name}")
            export_single(mutant, out_dir, seq_length=args.seq_length)
            exported += 1
        except Exception as e:
            print(f"  FAIL: {mutant.name}: {e}")
            failed += 1

    print("-" * 60)
    print(f"Done. Exported: {exported}, Failed: {failed}")
    print(f"Copy .onnx files to your MT5 terminal's MQL5\\Files\\ folder")


if __name__ == "__main__":
    main()

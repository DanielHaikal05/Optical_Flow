import argparse
import json
from pathlib import Path

import torch


def load_train_namespace(train_file, root):
    source = Path(train_file).read_text()
    definitions = source.split("model = Conv_Encoder_Decoder().to(\"cuda\")", 1)[0]
    namespace = {"__file__": str(train_file)}
    exec(compile(definitions, str(train_file), "exec"), namespace)
    namespace["ROOT"] = str(root)
    return namespace


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved NFlowNet checkpoint using train.py's evaluate method.")
    parser.add_argument("--train-file", default="train.py")
    parser.add_argument("--checkpoint", default="model_weights.pth")
    parser.add_argument("--root", default="/home/DanielH/Optical_Flow/Datasets/TartanAir")
    parser.add_argument("--output", default="evaluation_fixed_metrics.json")
    parser.add_argument("--sets", nargs="+", default=["Amusement", "Carwelding"])
    args = parser.parse_args()

    namespace = load_train_namespace(args.train_file, args.root)
    model = namespace["Conv_Encoder_Decoder"]().to("cuda")
    namespace["optimizer"] = torch.optim.Adam(model.parameters(), lr=namespace["LR"])

    state_dict = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(state_dict)

    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "root": str(Path(args.root).resolve()),
        "loss": {},
    }

    for dataset in args.sets:
        print(f"Evaluating on {dataset}")
        base = Path(args.root) / dataset
        loss = model.evaluate(
            base / "image_left",
            base / "normal_flow_gt" / "scalar",
            base / "normal_flow_gt" / "valid_mask",
            base / "normal_flow_gt" / "gradient_dir",
            dataset,
        )
        results["loss"][dataset] = loss

    output = Path(args.output)
    output.write_text(json.dumps(results, indent=4))
    print(json.dumps(results, indent=4))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

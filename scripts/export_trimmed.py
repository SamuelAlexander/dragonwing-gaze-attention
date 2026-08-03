"""Re-export EyeGaze without its `heatmaps` output.

EyeNet is a stacked hourglass, so it emits the per-stack heatmaps it used internally to
find the eye landmarks: (1, 3, 34, 48, 80) float32 = 1.57 MB per inference, per eye. An
application that wants gaze and landmarks never reads them, but the runtime still copies
them out of the delegate on every invoke.

This wraps the torch model to drop that output, then compiles the wrapper through AI Hub
the same way the zoo does. Run on the Mac, in the arm64 AI Hub venv.
"""

import argparse
import os

import qai_hub as hub
import torch
from qai_hub_models.utils.asset_loaders import always_answer_prompts

DEVICE = "Dragonwing IQ-9075 EVK"
# eyegaze/model.py adds these for every non-ONNX target; keep them so the trimmed
# build differs from the stock one in exactly one way.
COMPILE_OPTIONS = "--truncate_64bit_io --truncate_64bit_tensors"


class EyeGazeGazeOnly(torch.nn.Module):
    """EyeNet with the heatmaps output removed. Same weights, same maths."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, image):
        _heatmaps, landmarks, gaze_pitchyaw = self.base(image)
        return landmarks, gaze_pitchyaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/eyegaze_trimmed.tflite")
    args = ap.parse_args()

    with always_answer_prompts(True):
        from qai_hub_models.models.eyegaze.model import EyeGaze

        base = EyeGaze.from_pretrained()

    model = EyeGazeGazeOnly(base).eval()
    example = torch.zeros(1, 96, 160)

    with torch.no_grad():
        out = model(example)
    print(f"trimmed outputs: {[tuple(o.shape) for o in out]}")

    traced = torch.jit.trace(model, example)

    job = hub.submit_compile_job(
        model=traced,
        device=hub.Device(DEVICE),
        input_specs={"image": (1, 96, 160)},
        options=f"--target_runtime tflite {COMPILE_OPTIONS}",
    )
    print(f"compile job: {job.job_id}  ({job.url})")
    target = job.get_target_model()
    if target is None:
        raise SystemExit(f"compile failed: {job.get_status()}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    target.download(args.out)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

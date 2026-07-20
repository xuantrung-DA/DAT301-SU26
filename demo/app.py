"""Interactive report demo for the learned multi-domain UAV detector."""
from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference import load_detector, load_domain_router, route_domain_bgr  # noqa: E402

CHECKPOINTS = {
    "clean": ROOT / "runs/detect/runs/detector_b0_fix6/b0_seed_3407/weights/best.pt",
    "synthetic_lowlight": ROOT / "runs/detect/runs/detector_p2_fix5_finetune/b2_seed_3407/weights/best.pt",
    "real_lowlight": ROOT / "runs/detect/runs/domain_adaptation_exdark/b0_seed_3407-2/weights/best.pt",
}
ROUTER = ROOT / "runs/domain_router_final/best.pt"
DISPLAY = {"clean": "Clean / daylight", "synthetic_lowlight": "Synthetic low-light UAV", "real_lowlight": "Real low-light"}
FORCED = {"Auto (learned router)": None, **{f"Force: {label}": key for key, label in DISPLAY.items()}}


@lru_cache(maxsize=1)
def models():
    missing = [str(path) for path in [ROUTER, *CHECKPOINTS.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router, router_size = load_domain_router(ROUTER, device)
    detectors = {name: load_detector(str(path)) for name, path in CHECKPOINTS.items()}
    return device, router, router_size, detectors


def infer(image_rgb: np.ndarray | None, mode: str, confidence: float, iou: float):
    if image_rgb is None:
        raise gr.Error("Hãy upload hoặc chọn một ảnh demo.")
    device, router, router_size, detectors = models()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    routing = route_domain_bgr(router, image_bgr, device, router_size)
    router_ms = (time.perf_counter() - started) * 1000
    selected = FORCED[mode] or routing["domain_route"]
    detector_started = time.perf_counter()
    result = detectors[selected].predict(image_bgr, imgsz=640, conf=confidence, iou=iou, verbose=False)[0]
    if device.type == "cuda": torch.cuda.synchronize()
    detector_ms = (time.perf_counter() - detector_started) * 1000
    total_ms = (time.perf_counter() - started) * 1000
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    rows = []
    if result.boxes is not None:
        for class_id, score, box in zip(result.boxes.cls.cpu(), result.boxes.conf.cpu(), result.boxes.xyxy.cpu()):
            rows.append([result.names[int(class_id)], round(float(score), 3), *[round(float(value), 1) for value in box]])
    probabilities = routing["domain_probabilities"]
    status = (
        f"### Selected branch: `{DISPLAY[selected]}`\n"
        f"Router prediction: **{DISPLAY[routing['domain_route']]}**  \n"
        f"Clean `{probabilities['clean']:.1%}` · Synthetic LL `{probabilities['synthetic_lowlight']:.1%}` · Real LL `{probabilities['real_lowlight']:.1%}`\n\n"
        f"Detections: **{len(rows)}** · Router: **{router_ms:.2f} ms** · Detector: **{detector_ms:.2f} ms** · Total: **{total_ms:.2f} ms**"
    )
    telemetry = {
        "selected_branch": selected,
        "automatic_route": routing["domain_route"],
        "route_probabilities": probabilities,
        "detections": len(rows),
        "router_ms": router_ms,
        "detector_ms": detector_ms,
        "total_ms": total_ms,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    }
    return annotated, status, rows, telemetry


def examples() -> list[list[str]]:
    candidates = [
        ROOT / "datasets/VisDrone/images/val/0000001_02999_d_0000005.jpg",
        next(iter((ROOT / "datasets/VisDrone-LL/LL2/images/val").glob("*")), None),
        next(iter((ROOT / "datasets/ExDark-VisDroneOverlap/images/test").glob("*")), None),
    ]
    return [[str(path)] for path in candidates if path and path.is_file()]


CSS = """
.gradio-container {max-width: 1450px !important; margin: auto !important;}
.hero {background: linear-gradient(110deg,#071426,#132d46); padding:24px 30px; border-radius:18px; color:white;}
.hero h1 {margin:0 0 6px 0; font-size:30px;}
.metric {background:#f6f8fb; border:1px solid #dfe5ec; border-radius:12px; padding:10px;}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="LADD-UAV Learned Router Demo") as demo:
        gr.HTML("<div class='hero'><h1>LADD-UAV · Learned Conditional Routing</h1><div>One lightweight router → one specialist detector · Clean · Synthetic low-light · Real low-light</div></div>")
        with gr.Row():
            gr.Markdown("**LL2 mAP50**  \n## 0.1392", elem_classes="metric")
            gr.Markdown("**ExDark mAP50**  \n## 0.6049", elem_classes="metric")
            gr.Markdown("**Router balanced accuracy**  \n## 91.90%", elem_classes="metric")
            gr.Markdown("**Router p95**  \n## 0.523 ms", elem_classes="metric")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input UAV image", type="numpy", image_mode="RGB", height=520)
                mode = gr.Radio(list(FORCED), value="Auto (learned router)", label="Routing mode")
                with gr.Row():
                    confidence = gr.Slider(0.05, 0.8, 0.20, step=0.05, label="Confidence")
                    iou = gr.Slider(0.3, 0.9, 0.70, step=0.05, label="NMS IoU")
                run = gr.Button("Run detection", variant="primary", size="lg")
            with gr.Column(scale=1):
                output_image = gr.Image(label="Detection output", type="numpy", height=520)
                status = gr.Markdown("Upload an image, then run the learned router.")
        with gr.Tabs():
            with gr.Tab("Detections"):
                table = gr.Dataframe(headers=["Class", "Confidence", "x1", "y1", "x2", "y2"], datatype=["str", "number", "number", "number", "number", "number"], interactive=False)
            with gr.Tab("Telemetry"):
                telemetry = gr.JSON()
            with gr.Tab("How to present"):
                gr.Markdown("1. Run **Auto** on one clean, one LL2 and one ExDark image.\n2. Point out router probabilities and sub-millisecond overhead.\n3. Force a wrong branch to demonstrate why learned conditional routing matters.\n4. State clearly: ExDark specialist is supervised adaptation; routing is automatic, not zero-shot domain adaptation.")
        sample_rows = examples()
        if sample_rows:
            gr.Examples(sample_rows, inputs=[input_image], label="Local report samples")
        run.click(infer, [input_image, mode, confidence, iou], [output_image, status, table, telemetry])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=7860); parser.add_argument("--share", action="store_true"); parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(); build_demo().launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=not args.no_browser, css=CSS)


if __name__ == "__main__": main()

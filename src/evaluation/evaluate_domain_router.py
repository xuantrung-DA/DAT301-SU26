"""Evaluate learned routing and compare it with the luminance-threshold baseline."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.domain_router import DOMAIN_NAMES, LearnedDomainRouter
from training.train_domain_router import DomainDataset, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True)
    for domain in DOMAIN_NAMES: parser.add_argument(f"--{domain.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--batch", type=int, default=128)
    args=parser.parse_args(); state=torch.load(args.checkpoint,map_location="cpu",weights_only=False); size=int(state["size"])
    roots=[getattr(args,name) for name in DOMAIN_NAMES]; dataset=DomainDataset(roots,size,False); loader=DataLoader(dataset,batch_size=args.batch,shuffle=False,num_workers=0,pin_memory=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=LearnedDomainRouter().to(device); model.load_state_dict(state["model"]); model.eval()
    metrics=evaluate(model,loader,device); sample=torch.rand(1,3,size,size,device=device)
    for _ in range(100): model(sample)
    if device.type=="cuda": torch.cuda.synchronize()
    times=[]
    for _ in range(1000):
        start=time.perf_counter(); model(sample)
        if device.type=="cuda": torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
    metrics.update({"images":len(dataset),"parameters":sum(p.numel() for p in model.parameters()),"latency_mean_ms":float(np.mean(times)),"latency_p95_ms":float(np.percentile(times,95))})
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(metrics,indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2))

if __name__ == "__main__": main()

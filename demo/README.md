# Report demo

Start from PowerShell at the project root:

```powershell
.\scripts\run_demo.ps1
```

The browser opens at `http://127.0.0.1:7860`. Use `--share` only when an external temporary Gradio link is intentionally required.

The demo loads the learned router and all three final specialist checkpoints lazily on the first inference. The first run is slower because CUDA and model kernels warm up; subsequent runs represent the live presentation experience.

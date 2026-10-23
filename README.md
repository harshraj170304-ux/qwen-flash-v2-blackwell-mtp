# ⚡ Qwen3.8-Flash-Next v2 (180B Abliterated) + MTP on Blackwell

Fast 1-click deployment of **Huihui-AI Qwen3.8-Flash-Next v2 (Abliterated ~180B MoE)** with **Multi-Token Prediction (MTP)** and **Nous Research Hermes Agent** on an **NVIDIA RTX PRO 6000 Blackwell Server Edition (95.6 GB VRAM, `sm_120`)**.

---

## 🚀 1-Click Fast Deployment Command

Run this single command in your MoLab terminal (`/marimo#`):

```bash
curl -fsSL https://raw.githubusercontent.com/harshraj170304-ux/qwen-flash-v2-blackwell-mtp/main/setup.sh | bash
```

---

## 📊 Measured Hardware Benchmarks (2,048 Tokens Sustained)

| Metric | Without MTP | With MTP (Draft-MTP) | Speedup |
| :--- | :--- | :--- | :--- |
| **Sustained Throughput** | 58.45 tok/s | **86.94 tok/s** | **🚀 +48.7% faster** |
| **2K Generation Time** | 35.04 seconds | **23.56 seconds** | **⚡ 11.48s saved per response** |
| **Active Parameters** | ~6B active per token (512 MoE experts) | Shared 8-bit draft head | Speculative multi-token verify |
| **Context Window** | 262,144 Tokens | YaRN RoPE + q8_0 KV Cache | ~6.3 GB KV Cache footprint |

---

## 🛠️ Key Commands

| Task | Command |
| :--- | :--- |
| **Resume Hermes Agent Chat** | `hermes -c` or `hermes chat -c` |
| **Start Fresh Hermes Session** | `hermes-agent` |
| **List Saved Chat Sessions** | `hermes sessions list` |
| **Check Live Engine Generation** | `tmux attach -t qwen` *(Exit: `Ctrl+B`, `D`)* |
| **Check Token Generation Speed** | `tmux capture-pane -pt qwen -S -20` |
| **Force Immediate Cloud Memory Sync** | `hermes-sync` |
| **Open Interactive Chat Web UI** | Open `/marimo/app.py` in MoLab sidebar |

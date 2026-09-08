#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu GPU box (Lambda, RunPod, any cloud VM w/ NVIDIA driver) for this repo.
#
# Usage on a fresh instance:
#   git clone https://github.com/<you>/gpt-from-scratch.git && cd gpt-from-scratch
#   bash setup_gpu.sh
#
# Idempotent: safe to rerun. Needs a tty for gh auth (don't pipe into bash).

set -euo pipefail

cd "$(dirname "$0")"

# --- uv ---
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# --- gh cli (official apt repo) ---
if ! command -v gh >/dev/null; then
    sudo mkdir -p -m 755 /etc/apt/keyrings
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
    sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq gh
fi

# --- git identity: prompt if unset ---
if [ -z "$(git config --global user.name)" ]; then
    read -rp "git user.name: " name
    git config --global user.name "$name"
fi
if [ -z "$(git config --global user.email)" ]; then
    read -rp "git user.email: " email
    git config --global user.email "$email"
fi

# --- gh auth: browser device flow, https so no ssh key needed on the box ---
if ! gh auth status >/dev/null 2>&1; then
    gh auth login --hostname github.com --git-protocol https --web
fi
gh auth setup-git

# --- deps ---
uv sync

# --- sanity ---
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

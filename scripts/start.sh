#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   WANDB_API_KEY=... ./scripts/codex-start.sh
# or:
#   echo 'WANDB_API_KEY=...' > .env.codex
#   ./scripts/codex-start.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Optional local env file
if [[ -f .env.codex ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.codex
  set +a
fi

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your shell or .env.codex}"

install_uv() {
  echo "uv not found; installing..."

  local install_dir="${HOME}/.local/bin"
  mkdir -p "$install_dir"

  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$install_dir" sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$install_dir" sh
  else
    echo "error: need curl or wget to install uv" >&2
    exit 1
  fi

  export PATH="$install_dir:$PATH"

  if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv installation completed, but uv is still not on PATH" >&2
    echo "Try: export PATH=\"$install_dir:\$PATH\"" >&2
    exit 1
  fi
}

if ! command -v uv >/dev/null 2>&1; then
  install_uv
fi

uv venv

# Make the repo venv easy to inherit in this shell/session
export VIRTUAL_ENV="$REPO_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

source "$VIRTUAL_ENV/bin/activate"

# Install/update the repo environment
uv sync

mkdir -p .codex
CONFIG_FILE=".codex/config.toml"
touch "$CONFIG_FILE"

# Add W&B MCP config only if missing
if ! grep -q '^\[mcp_servers\.wandb\]$' "$CONFIG_FILE"; then
  cat >> "$CONFIG_FILE" <<'EOF'

[mcp_servers.wandb]
url = "https://mcp.withwandb.com/mcp"
bearer_token_env_var = "WANDB_API_KEY"
EOF
fi

echo "Done."
echo "uv: $(command -v uv)"
echo "venv: $VIRTUAL_ENV"
echo "codex config: $CONFIG_FILE"

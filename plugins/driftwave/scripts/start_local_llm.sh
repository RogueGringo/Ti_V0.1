#!/bin/bash
# Driftwave Local LLM Server — Adaptive Compute Selection
#
# Auto-detects GPU availability and selects the optimal model:
#   GPU FREE  → Llama 3.2 3B on CUDA (transformer, most capable)
#   GPU BUSY  → LFM 1.2 on CPU (state-space model, fast linear inference)
#   NO GPU    → LFM 1.2 on CPU (default fallback)
#
# Override with: DW_LLM_DEVICE=cpu|cuda DW_LLM_MODEL=llama|lfm
#
# Axiom: ADAPTIVE_SCALE — compute allocation adapts to hardware state

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PLUGIN_ROOT/models"
PORT="${DW_LLM_PORT:-8090}"

# Colors
AMBER='\033[0;33m'
TEAL='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
FAINT='\033[2m'
NC='\033[0m'

# Model paths
LLAMA_MODEL="$MODEL_DIR/llama-3.2-3b-instruct-q4_k_m.gguf"
LFM_MODEL="$MODEL_DIR/lfm-1.2-1b-q4_k_m.gguf"

echo -e "${AMBER}[driftwave]${NC} Local LLM — Adaptive Compute Selection"
echo -e "${FAINT}  Axiom: ADAPTIVE_SCALE — hardware state determines model routing${NC}"

# ─── Check if already running ───
if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
  RUNNING_MODEL=$(curl -s "http://localhost:$PORT/v1/models" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',[{}])[0].get('id','unknown'))" 2>/dev/null || echo "unknown")
  echo -e "${GREEN}[driftwave]${NC} Server already running on port $PORT (model: $RUNNING_MODEL)"
  exit 0
fi

# ─── Auto-detect compute state ───
detect_device() {
  # User override
  if [ -n "$DW_LLM_DEVICE" ]; then
    echo "$DW_LLM_DEVICE"
    return
  fi

  # Check if CUDA is available
  if ! command -v nvidia-smi &>/dev/null; then
    echo "cpu"
    return
  fi

  # Check if GPU has active compute processes
  GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')

  if [ "$GPU_PROCS" -gt 0 ]; then
    # GPU busy — check VRAM headroom
    VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo -e "${TEAL}[driftwave]${NC} GPU has $GPU_PROCS active process(es), ${VRAM_FREE}MB free VRAM" >&2

    # Need at least 2500MB for Llama 3B quantized
    if [ "${VRAM_FREE:-0}" -ge 2500 ]; then
      echo -e "${TEAL}[driftwave]${NC} Enough VRAM headroom — can share GPU" >&2
      echo "cuda"
    else
      echo -e "${TEAL}[driftwave]${NC} Insufficient VRAM headroom — routing to CPU" >&2
      echo "cpu"
    fi
  else
    echo "cuda"
  fi
}

select_model() {
  local device="$1"

  # User override
  if [ -n "$DW_LLM_MODEL" ]; then
    case "$DW_LLM_MODEL" in
      llama) echo "llama" ;;
      lfm)   echo "lfm" ;;
      *)     echo "llama" ;;
    esac
    return
  fi

  # Auto-select based on device
  if [ "$device" = "cuda" ]; then
    # GPU path: prefer Llama (transformer, more capable with GPU acceleration)
    if [ -f "$LLAMA_MODEL" ]; then
      echo "llama"
    elif [ -f "$LFM_MODEL" ]; then
      echo "lfm"
    else
      echo "none"
    fi
  else
    # CPU path: prefer LFM (state-space model, O(n) linear, fast on CPU)
    if [ -f "$LFM_MODEL" ]; then
      echo "lfm"
    elif [ -f "$LLAMA_MODEL" ]; then
      echo "llama"
    else
      echo "none"
    fi
  fi
}

DEVICE=$(detect_device)
MODEL_CHOICE=$(select_model "$DEVICE")

echo -e "${AMBER}[driftwave]${NC} Device: $DEVICE"
echo -e "${AMBER}[driftwave]${NC} Model: $MODEL_CHOICE"

# ─── Resolve model path and GPU layers ───
case "$MODEL_CHOICE" in
  llama)
    MODEL_PATH="${DW_LLM_MODEL_PATH:-$LLAMA_MODEL}"
    MODEL_NAME="Llama 3.2 3B Instruct (Q4_K_M)"
    if [ "$DEVICE" = "cuda" ]; then
      GPU_LAYERS="-1"  # all layers on GPU
    else
      GPU_LAYERS="0"   # all layers on CPU
    fi
    CTX_SIZE=4096
    ;;
  lfm)
    MODEL_PATH="${DW_LLM_MODEL_PATH:-$LFM_MODEL}"
    MODEL_NAME="LFM 1.2 1B (Q4_K_M)"
    if [ "$DEVICE" = "cuda" ]; then
      GPU_LAYERS="-1"
    else
      GPU_LAYERS="0"   # SSM is fast on CPU — this is the sweet spot
    fi
    CTX_SIZE=8192  # LFM supports longer context efficiently (linear scaling)
    ;;
  none)
    echo -e "${RED}[driftwave]${NC} No model found in $MODEL_DIR"
    echo ""
    echo "Download one or both models:"
    echo ""
    echo "  # Llama 3.2 3B (transformer, best on GPU):"
    echo "  huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \\"
    echo "    --include 'Llama-3.2-3B-Instruct-Q4_K_M.gguf' \\"
    echo "    --local-dir $MODEL_DIR"
    echo ""
    echo "  # LFM 1.2 (state-space, best on CPU):"
    echo "  huggingface-cli download LiquidAI/LFM-1.2-1B-GGUF \\"
    echo "    --include 'lfm-1.2-1b-q4_k_m.gguf' \\"
    echo "    --local-dir $MODEL_DIR"
    echo ""
    echo "Or override: DW_LLM_MODEL_PATH=/path/to/any.gguf $0"
    exit 1
    ;;
esac

# ─── Verify model file exists ───
if [ ! -f "$MODEL_PATH" ]; then
  echo -e "${RED}[driftwave]${NC} Model file not found: $MODEL_PATH"
  echo "Run the download command above, or set DW_LLM_MODEL_PATH"
  exit 1
fi

# ─── Check runtime ───
if ! python3 -c "import llama_cpp" 2>/dev/null; then
  echo -e "${RED}[driftwave]${NC} llama-cpp-python not installed"
  if [ "$DEVICE" = "cuda" ]; then
    echo "Install with CUDA: pip install 'llama-cpp-python[server]' --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128"
  else
    echo "Install CPU-only: pip install 'llama-cpp-python[server]'"
  fi
  exit 1
fi

# ─── Launch ───
echo -e "${AMBER}[driftwave]${NC} ─────────────────────────────────────"
echo -e "${AMBER}[driftwave]${NC} Model:  $MODEL_NAME"
echo -e "${AMBER}[driftwave]${NC} File:   $(basename "$MODEL_PATH")"
echo -e "${AMBER}[driftwave]${NC} Device: $DEVICE (gpu_layers=$GPU_LAYERS)"
echo -e "${AMBER}[driftwave]${NC} Port:   $PORT"
echo -e "${AMBER}[driftwave]${NC} CTX:    $CTX_SIZE"
echo -e "${AMBER}[driftwave]${NC} ─────────────────────────────────────"

nohup python3 -m llama_cpp.server \
  --model "$MODEL_PATH" \
  --n_gpu_layers "$GPU_LAYERS" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --n_ctx "$CTX_SIZE" \
  > "$MODEL_DIR/llm_server.log" 2>&1 &

LLM_PID=$!
echo "$LLM_PID" > "$MODEL_DIR/llm_server.pid"
echo "$DEVICE" > "$MODEL_DIR/llm_server.device"
echo "$MODEL_CHOICE" > "$MODEL_DIR/llm_server.model"

# ─── Wait for readiness ───
echo -e "${AMBER}[driftwave]${NC} Waiting for server (PID $LLM_PID)..."
TIMEOUT=30
if [ "$DEVICE" = "cpu" ]; then
  TIMEOUT=45  # CPU model load is slower
fi

for i in $(seq 1 "$TIMEOUT"); do
  if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo -e "${GREEN}[driftwave]${NC} ✓ Server ready on http://localhost:$PORT"
    echo -e "${GREEN}[driftwave]${NC} ✓ OpenAI-compatible API at http://localhost:$PORT/v1"
    echo -e "${GREEN}[driftwave]${NC} ✓ $MODEL_NAME on $DEVICE"
    echo ""
    echo -e "${FAINT}  Stop: kill \$(cat $MODEL_DIR/llm_server.pid)${NC}"
    echo -e "${FAINT}  Logs: tail -f $MODEL_DIR/llm_server.log${NC}"
    echo -e "${FAINT}  Force GPU: DW_LLM_DEVICE=cuda DW_LLM_MODEL=llama $0${NC}"
    echo -e "${FAINT}  Force CPU: DW_LLM_DEVICE=cpu DW_LLM_MODEL=lfm $0${NC}"
    exit 0
  fi
  sleep 1
done

echo -e "${RED}[driftwave]${NC} Server failed to start within ${TIMEOUT}s"
echo "Check logs: tail -20 $MODEL_DIR/llm_server.log"
exit 1

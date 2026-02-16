#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON_CMD="${PYTHON_CMD:-python3.11}"

if ! command -v "$PYTHON_CMD" &>/dev/null; then
    echo "ERROR: $PYTHON_CMD не найден. Установите Python 3.11 или задайте PYTHON_CMD."
    exit 1
fi

PY_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PY_VERSION" != "3.11" ]]; then
    echo "WARNING: Ожидался Python 3.11, найден $PY_VERSION"
fi

# --- Создание виртуального окружения ---
if [ ! -d "venv" ]; then
    echo ">>> Создание venv..."
    "$PYTHON_CMD" -m venv venv
fi
source venv/bin/activate

echo ">>> Обновление pip..."
pip install --upgrade pip

echo ">>> Установка зависимостей..."
pip install -r requirements.txt

echo ">>> Установка piper-phonemize (из k2-fsa зеркала)..."
pip install piper-phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

# --- Клонирование piper-sample-generator ---
if [ ! -d "piper-sample-generator" ]; then
    echo ">>> Клонирование piper-sample-generator..."
    git clone https://github.com/rhasspy/piper-sample-generator
fi

PIPER_MODEL="piper-sample-generator/models/en_US-libritts_r-medium.pt"
if [ ! -f "$PIPER_MODEL" ]; then
    echo ">>> Скачивание TTS-модели..."
    mkdir -p "piper-sample-generator/models"
    curl -L -o "$PIPER_MODEL" \
        "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"
fi

# --- Скачивание моделей openwakeword ---
OWW_MODELS_DIR=$(python -c "import openwakeword; import os; print(os.path.join(os.path.dirname(openwakeword.__file__), 'resources', 'models'))")
mkdir -p "$OWW_MODELS_DIR"

OWW_BASE_URL="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

for MODEL_FILE in embedding_model.onnx embedding_model.tflite melspectrogram.onnx melspectrogram.tflite; do
    DEST="$OWW_MODELS_DIR/$MODEL_FILE"
    if [ ! -f "$DEST" ]; then
        echo ">>> Скачивание $MODEL_FILE..."
        curl -L -o "$DEST" "$OWW_BASE_URL/$MODEL_FILE"
    fi
done

echo ""
echo "=== Установка завершена ==="
echo "Активируйте окружение: source venv/bin/activate"

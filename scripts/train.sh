#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-config/default.yaml}"

if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: venv не найден. Сначала запустите: bash scripts/setup.sh"
    exit 1
fi

source venv/bin/activate

echo ">>> Конфигурация: $CONFIG"
echo ""

# --- Шаг 1: Генерация синтетических клипов ---
echo "=== Шаг 1/3: Генерация клипов ==="
python train.py --training_config "$CONFIG" --generate_clips

# --- Шаг 2: Аугментация клипов ---
echo "=== Шаг 2/3: Аугментация ==="
python train.py --training_config "$CONFIG" --augment_clips

# --- Шаг 3: Обучение модели ---
echo "=== Шаг 3/3: Обучение модели ==="
python train.py --training_config "$CONFIG" --train_model

echo ""
echo "=== Обучение завершено ==="
echo "Модель сохранена в директорию, указанную в конфиге (output_dir)"

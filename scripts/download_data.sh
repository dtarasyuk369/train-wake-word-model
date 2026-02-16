#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Парсинг аргументов ---
NO_VENV=false
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-venv)
            NO_VENV=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Использование: $0 [опции]"
            echo ""
            echo "Опции:"
            echo "  --no-venv       Не активировать виртуальное окружение"
            echo "  --data-dir DIR  Директория для данных (по умолчанию: ./data)"
            echo "  -h, --help      Показать эту справку"
            echo ""
            echo "Переменные окружения:"
            echo "  DATA_DIR        Директория для данных"
            echo "  INCLUDE_FMA     Если 'true', скачать FMA (~7 ГБ)"
            exit 0
            ;;
        *)
            echo "Неизвестная опция: $1"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

mkdir -p "$DATA_DIR"

# --- Pre-computed openwakeword features ---
FEATURES_FILE="$DATA_DIR/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
if [ ! -f "$FEATURES_FILE" ]; then
    echo ">>> Скачивание ACAV100M features..."
    curl -L -o "$FEATURES_FILE" \
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy?download=true"
else
    echo ">>> ACAV100M features уже скачаны, пропуск"
fi

VAL_FILE="$DATA_DIR/validation_set_features.npy"
if [ ! -f "$VAL_FILE" ]; then
    echo ">>> Скачивание validation set features..."
    curl -L -o "$VAL_FILE" \
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy?download=true"
else
    echo ">>> Validation features уже скачаны, пропуск"
fi

# --- Активация venv если нужно ---
if [ "$NO_VENV" = false ]; then
    if [ ! -f "$PROJECT_DIR/venv/bin/activate" ]; then
        echo "ERROR: venv не найден. Сначала запустите: bash scripts/setup.sh"
        exit 1
    fi
    source "$PROJECT_DIR/venv/bin/activate"
    echo ">>> Виртуальное окружение активировано"
else
    echo ">>> Режим --no-venv: используется системный Python"
fi

# --- Python-часть: MIT RIRs, AudioSet 16kHz, (опционально FMA) ---
echo ">>> Запуск download_data.py (MIT RIRs, AudioSet 16kHz)..."

FMA_FLAG=""
if [ "${INCLUDE_FMA:-false}" = "true" ]; then
    FMA_FLAG="--include-fma"
    echo ">>> FMA включён (INCLUDE_FMA=true)"
fi

python "$PROJECT_DIR/download_data.py" --data-dir "$DATA_DIR" $FMA_FLAG

echo ""
echo "=== Загрузка данных завершена ==="
echo "Данные в: $DATA_DIR"
if [ "$NO_VENV" = false ]; then
    echo "Для загрузки FMA (~7 ГБ): INCLUDE_FMA=true bash scripts/download_data.sh"
else
    echo "Для загрузки FMA (~7 ГБ): INCLUDE_FMA=true bash scripts/download_data.sh --no-venv"
fi

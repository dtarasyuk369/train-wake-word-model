#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Парсинг аргументов ---
NO_VENV=false
CONFIG="config/default.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-venv)
            NO_VENV=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Использование: $0 [опции]"
            echo ""
            echo "Опции:"
            echo "  --no-venv         Не активировать виртуальное окружение"
            echo "  --config FILE     Путь к конфигурации (по умолчанию: config/default.yaml)"
            echo "  -h, --help        Показать эту справку"
            echo ""
            echo "Примеры:"
            echo "  bash scripts/train.sh"
            echo "  bash scripts/train.sh --config config/my_model.yaml"
            echo "  bash scripts/train.sh --no-venv --config config/colab_config.yaml"
            exit 0
            ;;
        *)
            # Если аргумент без флага - считаем его путём к конфигу (обратная совместимость)
            if [ -f "$1" ]; then
                CONFIG="$1"
                shift
            else
                echo "Неизвестная опция или файл не найден: $1"
                echo "Используйте --help для справки"
                exit 1
            fi
            ;;
    esac
done

# --- Проверка конфига ---
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Конфигурация не найдена: $CONFIG"
    exit 1
fi

# --- Активация venv если нужно ---
if [ "$NO_VENV" = false ]; then
    if [ ! -f "venv/bin/activate" ]; then
        echo "ERROR: venv не найден. Сначала запустите: bash scripts/setup.sh"
        exit 1
    fi
    source venv/bin/activate
    echo ">>> Виртуальное окружение активировано"
else
    echo ">>> Режим --no-venv: используется системный Python"
fi

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

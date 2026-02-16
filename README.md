# 🎙️ Train Wake Word Model

Проект для обучения собственных моделей пробуждения (wake word detection) с использованием [OpenWakeWord](https://github.com/dscripka/openWakeWord) и синтетической генерации данных через [Piper TTS](https://github.com/rhasspy/piper).

## 🚀 Возможности

- ✅ Обучение пользовательских моделей для любого слова/фразы
- ✅ Синтетическая генерация обучающих данных с TTS
- ✅ Автоматическая аугментация данных (шум, реверберация)
- ✅ Поддержка Google Colab (бесплатное GPU обучение)
- ✅ Экспорт моделей в формат ONNX
- ✅ Низкие требования к ресурсам для inference

## 📋 Требования

### Локальная установка

- Python 3.8+
- NVIDIA GPU с CUDA (рекомендуется, но не обязательно)
- 10+ ГБ свободного места на диске
- 8+ ГБ RAM

### Google Colab

- Аккаунт Google
- Стабильное интернет-соединение
- ~2-8 часов для полного обучения

## 🛠️ Установка

### Вариант 1: Google Colab (рекомендуется для начинающих)

1. Откройте ноутбук `colab_train.ipynb` в [Google Colab](https://colab.research.google.com/)
2. Следуйте инструкциям в ноутбуке
3. Подробная документация: [COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md)

### Вариант 2: Локальная установка

```bash
# Клонируйте репозиторий
git clone https://github.com/YOUR_USERNAME/train-wake-word-model.git
cd train-wake-word-model

# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установите зависимости
pip install -r requirements.txt

# Клонируйте Piper Sample Generator (для генерации TTS)
git clone https://github.com/rhasspy/piper-sample-generator.git
cd piper-sample-generator
pip install -e .
cd ..
```

## 📚 Быстрый старт

### 1. Скачайте данные

```bash
# Базовые данные (MIT RIRs + AudioSet)
python download_data.py

# Опционально: добавить музыкальный фон (~7 ГБ)
python download_data.py --include-fma --fma-hours 2
```

### 2. Настройте конфигурацию

Отредактируйте `config/default.yaml`:

```yaml
model_name: "my_wake_word"

target_phrase:
  - "hey assistant"  # Ваше пробуждающее слово

n_samples: 5000  # Количество обучающих примеров
steps: 10000     # Шаги обучения
```

### 3. Обучите модель

```bash
# Полный пайплайн (генерация → аугментация → обучение)
python train.py \
  --training_config config/default.yaml \
  --generate_clips \
  --augment_clips \
  --train_model
```

Или пошагово:

```bash
# Шаг 1: Генерация синтетических образцов
python train.py --training_config config/default.yaml --generate_clips

# Шаг 2: Аугментация данных
python train.py --training_config config/default.yaml --augment_clips

# Шаг 3: Обучение модели
python train.py --training_config config/default.yaml --train_model
```

### 4. Получите модель

Обученная модель будет сохранена в:
```
./my_custom_model/your_model_name.onnx
```

## 📖 Использование модели

### Python

```python
from openwakeword.model import Model
import pyaudio
import numpy as np

# Загрузить модель
owwModel = Model(wakeword_models=["./my_custom_model/my_wake_word.onnx"])

# Настроить микрофон
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

audio = pyaudio.PyAudio()
mic_stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

# Слушать и детектировать
print("Listening for wake word...")
while True:
    audio_data = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
    prediction = owwModel.predict(audio_data)

    for mdl_name, score in prediction.items():
        if score > 0.5:
            print(f"✅ Wake word detected! Score: {score:.2f}")
```

### Интеграция в приложение

```python
import openwakeword
from openwakeword.model import Model

class WakeWordDetector:
    def __init__(self, model_path, threshold=0.5):
        self.model = Model(wakeword_models=[model_path])
        self.threshold = threshold

    def detect(self, audio_chunk):
        """
        audio_chunk: numpy array (1280 samples @ 16kHz = 80ms)
        Returns: True if wake word detected
        """
        predictions = self.model.predict(audio_chunk)
        return any(score > self.threshold for score in predictions.values())

# Использование
detector = WakeWordDetector("./my_custom_model/my_wake_word.onnx")
# ... получить audio_chunk из микрофона
if detector.detect(audio_chunk):
    print("Wake word detected!")
```

## ⚙️ Настройка параметров

### Основные параметры

| Параметр | Описание | Рекомендуемые значения |
|----------|----------|------------------------|
| `n_samples` | Количество обучающих примеров | 3000-10000 |
| `n_samples_val` | Количество валидационных примеров | 1000-5000 |
| `steps` | Количество шагов обучения | 5000-20000 |
| `layer_size` | Размер скрытых слоёв модели | 32-128 |
| `max_negative_weight` | Вес негативных примеров | 1000-2000 |
| `target_false_positives_per_hour` | Целевая частота ложных срабатываний | 0.1-0.5 |

### Компромиссы

**Больше примеров и шагов** → Лучше качество, но дольше обучение

**Больше `max_negative_weight`** → Меньше ложных срабатываний, но может пропускать настоящие

**Меньше `layer_size`** → Быстрее inference, но может быть менее точным

## 📁 Структура проекта

```
train-wake-word-model/
├── config/
│   └── default.yaml              # Конфигурация обучения
├── data/
│   ├── audioset_16k/             # Фоновый шум
│   ├── mit_rirs/                 # Импульсные характеристики
│   └── *.npy                     # Предобработанные признаки
├── my_custom_model/
│   └── your_model/
│       ├── positive_train/       # Позитивные образцы для обучения
│       ├── negative_train/       # Негативные образцы для обучения
│       └── your_model.onnx       # Обученная модель
├── piper-sample-generator/       # TTS генератор
├── scripts/
│   ├── download_data.sh          # Скрипт скачивания данных
│   ├── setup.sh                  # Скрипт установки
│   └── train.sh                  # Скрипт обучения
├── colab_train.ipynb             # Ноутбук для Google Colab
├── download_data.py              # Скачивание датасетов
├── train.py                      # Основной скрипт обучения
├── requirements.txt              # Зависимости Python
├── COLAB_INSTRUCTIONS.md         # Инструкция для Colab
└── README.md                     # Этот файл
```

## 🐛 Устранение проблем

### CUDA out of memory

Уменьшите размеры батчей:
```yaml
tts_batch_size: 16
augmentation_batch_size: 8
batch_n_per_class:
  ACAV100M_sample: 512
  adversarial_negative: 32
  positive: 32
```

### Низкая точность модели

1. Увеличьте `n_samples` (больше обучающих примеров)
2. Увеличьте `steps` (дольше обучение)
3. Добавьте больше разнообразного фонового шума
4. Увеличьте `augmentation_rounds`
5. Используйте `layer_size: 64` или `128`

### Много ложных срабатываний

Увеличьте `max_negative_weight` или `target_false_positives_per_hour: 0.1`

### Модель пропускает активации

Уменьшите `max_negative_weight` или добавьте больше позитивных примеров

## 🌐 Поддержка языков

По умолчанию проект использует английскую TTS модель. Для других языков:

1. Скачайте модель Piper для нужного языка: https://github.com/rhasspy/piper/releases
2. Укажите путь в конфигурации:
```yaml
piper_model_path: "./models/ru_RU-ruslan-medium.pt"  # Пример для русского
```

Доступные языки: английский, русский, немецкий, французский, испанский, китайский и [другие](https://github.com/rhasspy/piper/blob/master/VOICES.md).

## 📊 Производительность

Типичные результаты на тестовых данных:

- **Accuracy**: 85-95%
- **Recall**: 70-90%
- **False Positives**: 0.1-0.5 per hour
- **Latency**: ~50ms на CPU, ~10ms на GPU
- **Model Size**: 50-200 KB (ONNX)

## 🤝 Вклад в проект

Contributions welcome! Пожалуйста:

1. Fork репозиторий
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📄 Лицензия

Этот проект использует следующие open-source проекты:

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) - Apache 2.0
- [Piper TTS](https://github.com/rhasspy/piper) - MIT
- [PyTorch](https://pytorch.org/) - BSD

## 🙏 Благодарности

- [David Scripka](https://github.com/dscripka) за OpenWakeWord
- [Rhasspy](https://github.com/rhasspy) за Piper TTS
- [AudioSet](https://research.google.com/audioset/) и [MIT Acoustical Reverberation Scene Statistics Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) за датасеты

## 📞 Поддержка

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/train-wake-word-model/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/train-wake-word-model/discussions)
- **Email**: your.email@example.com

## 🗺️ Roadmap

- [ ] Поддержка многоклассовых моделей
- [ ] Web UI для обучения
- [ ] Предобученные модели для популярных слов
- [ ] Поддержка edge-устройств (Raspberry Pi, ESP32)
- [ ] Real-time мониторинг обучения
- [ ] Автоматический подбор гиперпараметров

---

Made with ❤️ for voice interface developers

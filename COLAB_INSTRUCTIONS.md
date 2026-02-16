# Инструкция по использованию в Google Colab

## Быстрый старт

### 1. Откройте ноутбук в Colab

1. Перейдите на [Google Colab](https://colab.research.google.com/)
2. Нажмите **File → Upload notebook** (Файл → Загрузить ноутбук)
3. Загрузите файл `colab_train.ipynb` из этого репозитория
4. Или используйте прямую ссылку: `https://colab.research.google.com/github/YOUR_USERNAME/train-wake-word-model/blob/main/colab_train.ipynb`

### 2. Включите GPU

⚠️ **Важно**: Для обучения требуется GPU!

1. В меню Colab выберите: **Runtime → Change runtime type** (Среда выполнения → Сменить тип среды)
2. В поле **Hardware accelerator** выберите **GPU** (T4, V100 или A100)
3. Нажмите **Save** (Сохранить)

### 3. Выполните ячейки по порядку

Запускайте ячейки последовательно, начиная с первой:

#### Ячейка 1: Установка окружения
```python
!nvidia-smi  # Проверка GPU
!git clone https://github.com/YOUR_USERNAME/train-wake-word-model.git
%cd train-wake-word-model
!pip install -q -r requirements.txt
```

#### Ячейка 2: Скачивание данных
```python
!python download_data.py --data-dir ./data --audioset-clips 3000
```
⏱️ Займёт ~10-15 минут

#### Ячейка 3: Настройка конфигурации
Измените параметры под свои нужды:
- `model_name`: имя модели (например, `'my_wake_word'`)
- `target_phrase`: ваше пробуждающее слово (например, `['hey assistant']`)
- `n_samples`: количество образцов для обучения (рекомендуется 3000-5000 для Colab)
- `steps`: количество шагов обучения (рекомендуется 5000-10000 для Colab)

#### Ячейка 4: Генерация синтетических образцов
```python
!python train.py --training_config ./config/colab_config.yaml --generate_clips
```
⏱️ Займёт ~20-40 минут

#### Ячейка 5: Аугментация данных
```python
!python train.py --training_config ./config/colab_config.yaml --augment_clips
```
⏱️ Займёт ~15-30 минут

#### Ячейка 6: Обучение модели
```python
!python train.py --training_config ./config/colab_config.yaml --train_model
```
⏱️ Займёт ~30-90 минут в зависимости от параметров

#### Ячейка 7: Скачивание модели
Скачайте обученную модель `.onnx` на ваш компьютер

---

## Советы и рекомендации

### Ограничения Google Colab

- **Время сессии**: Бесплатная версия ограничена ~12 часами непрерывной работы
- **GPU время**: Ограничено ~15-20 часами в неделю для бесплатных пользователей
- **Диск**: ~100 ГБ временного хранилища (удаляется после закрытия сессии)
- **RAM**: 12-13 ГБ для бесплатной версии

### Оптимизация для Colab

Чтобы уложиться в лимиты, используйте следующие параметры:

**Быстрое обучение (1-2 часа)**:
```yaml
n_samples: 2000
n_samples_val: 500
steps: 3000
augmentation_rounds: 1
```

**Сбалансированное обучение (3-4 часа)**:
```yaml
n_samples: 3000
n_samples_val: 1000
steps: 5000
augmentation_rounds: 1
```

**Качественное обучение (5-8 часов)**:
```yaml
n_samples: 5000
n_samples_val: 2000
steps: 10000
augmentation_rounds: 2
```

### Сохранение прогресса

Если сессия прервалась, вы можете:

1. **Смонтировать Google Drive** для сохранения данных:
```python
from google.colab import drive
drive.mount('/content/drive')

# Измените output_dir в конфиге:
config['output_dir'] = '/content/drive/MyDrive/wake_word_models'
```

2. **Скачать промежуточные результаты**:
```python
from google.colab import files
files.download('./my_custom_model.zip')
```

### Мониторинг ресурсов

Следите за использованием ресурсов в Colab:
- GPU: Нажмите на иконку RAM/Disk в правом верхнем углу
- Логи обучения: Смотрите вывод в ячейке обучения

### Устранение проблем

**Проблема**: "CUDA out of memory"
**Решение**:
```yaml
# Уменьшите размеры батчей:
tts_batch_size: 16
augmentation_batch_size: 8
batch_n_per_class:
  ACAV100M_sample: 512
  adversarial_negative: 32
  positive: 32
```

**Проблема**: "Session timeout"
**Решение**: Используйте Google Drive для сохранения прогресса или разбейте процесс на части

**Проблема**: "Модель не скачивается"
**Решение**: Проверьте путь к модели:
```python
!ls -la ./my_custom_model/
```

### Альтернативы Colab

Если лимиты Colab недостаточны:

1. **Kaggle Notebooks**: 30 часов GPU/неделю, 16 ГБ RAM
2. **Google Colab Pro**: $10/месяц, больше GPU времени и RAM
3. **Paperspace Gradient**: Бесплатные GPU ноутбуки
4. **Локальная машина**: Если у вас есть NVIDIA GPU

---

## Использование обученной модели

После скачивания `.onnx` файла:

### Python
```python
from openwakeword.model import Model

# Загрузить модель
owwModel = Model(wakeword_models=["path/to/your_model.onnx"])

# Использовать для детекции
import pyaudio
import numpy as np

# Настройка микрофона
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

print("Listening...")
while True:
    audio_data = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
    prediction = owwModel.predict(audio_data)

    for mdl_name, score in prediction.items():
        if score > 0.5:  # Порог детекции
            print(f"Detected: {mdl_name} (score: {score})")
```

### С microphone package
```python
import openwakeword
from openwakeword.model import Model
import pyaudio

owwModel = Model(wakeword_models=["your_model.onnx"])

# Непрерывное прослушивание
for prediction in owwModel.predict_clip(mic_source='microphone'):
    for mdl, score in prediction.items():
        if score > 0.5:
            print(f"Wake word detected: {mdl}")
```

---

## Часто задаваемые вопросы

**Q: Сколько времени занимает полное обучение?**
A: От 2 до 8 часов в зависимости от параметров и доступного GPU.

**Q: Можно ли обучить модель на русском языке?**
A: Да, но потребуется русская TTS модель для Piper. Замените `piper_model_path` на русскую модель.

**Q: Как улучшить качество модели?**
A: Увеличьте `n_samples`, `steps`, добавьте больше фонового шума, используйте больше `augmentation_rounds`.

**Q: Модель работает плохо, что делать?**
A:
1. Увеличьте количество обучающих примеров
2. Добавьте больше разнообразного фонового шума
3. Увеличьте количество шагов обучения
4. Настройте `max_negative_weight` для баланса false positives/false negatives

**Q: Можно ли использовать несколько пробуждающих слов?**
A: Да, добавьте их в список `target_phrase`:
```yaml
target_phrase:
  - "hey assistant"
  - "wake up"
  - "hello computer"
```

---

## Поддержка

- **GitHub Issues**: [создайте issue](https://github.com/YOUR_USERNAME/train-wake-word-model/issues)
- **OpenWakeWord документация**: https://github.com/dscripka/openWakeWord
- **Piper TTS**: https://github.com/rhasspy/piper

---

## Лицензия

Этот проект распространяется под лицензией, указанной в основном репозитории.

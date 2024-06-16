# Вычислительно эффективная модель для визуальной классикации произнесенных на видео слов

В данном репозитории содержится реализация [статьи](https://www.researchgate.net/publication/360794222_Accurate_and_Resource-Efficient_Lipreading_with_Efficientnetv2_and_Transformers?enrichId=rgreq-f863c780260845c92418bc773235cd22-XXX&enrichSource=Y292ZXJQYWdlOzM2MDc5NDIyMjtBUzoxMTQzMTI4MTA5NTk5NTA2N0AxNjY4MDY3MjI5NDM5&el=1_x_2&_esc=publicationCoverPdf) . До этого открытый код модели отсутстовал, поэтому мной была реализована вся архитектура и логика обучения, валидации и инференса с нуля.

Модель предсказывает логарифм вероятности принадлежности произнесенного на видео слова к одному из 500 классов, представленных в датасете LRW.

## Установка зависимостей
Для того, чтобы установить среду разработки, выполните следующие шаги.

### Создайте и активируйте venv
Создание
```bash
python3 -m venv venv
```
Активация
```bash
source venv/bin/activate
```

### Соберите среду
```bash
pip install .
```

### Обновите hydra
Это нужно для корректной работы
```bash
pip install hydra-core --upgrade
```
### Поставьте gale
Он нужен для обработчика EMA 
```bash
git clone https://github.com/benihime91/gale
cd gale
pip install .
cd ..
```


## Реализованная архитектура
Архитектура реализованной модели представлена ниже.
![photo_2024-06-13_00-52-23](https://github.com/sadevans/EfLipSystem/assets/82286355/97dcf13e-f5d0-48e7-89b5-1869628d7248)

Модель состоит из блока 3D сверточной сети, отмасштабированной EfficientNetV2, энкодера трансформера и блока временной сверточной сети (TCN). Розовым на рисунке обозначена внешняя часть сети (frontend), выполняющая извлечение признакв, оранжевым - внутренняя часть сети (backend), отвечающая за обработку признаков.

## Обучение модели
Логика программного комплекса обучения и валидации модели представлена ниже.

![program_train](https://github.com/sadevans/EfLipReading/assets/82286355/45050acc-2723-4d0b-b673-11452c05e5ea)

Для обучения используется `pytorch-lightning`. Для загрузки конфигураций - `hydra`.

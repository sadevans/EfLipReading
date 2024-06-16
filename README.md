# Вычислительно эффективная модель для визуальной классикации произнесенных на видео слов

В данном репозитории содержится реализация [статьи](https://www.researchgate.net/publication/360794222_Accurate_and_Resource-Efficient_Lipreading_with_Efficientnetv2_and_Transformers?enrichId=rgreq-f863c780260845c92418bc773235cd22-XXX&enrichSource=Y292ZXJQYWdlOzM2MDc5NDIyMjtBUzoxMTQzMTI4MTA5NTk5NTA2N0AxNjY4MDY3MjI5NDM5&el=1_x_2&_esc=publicationCoverPdf) . До этого открытый код модели отсутстовал, поэтому мной была реализована вся архитектура и логика обучения, валидации и инференса с нуля.

Модель предсказывает логарифм вероятности принадлежности произнесенного на видео слова к одному из 500 классов, представленных в датасете LRW.

## Установка зависимостей



## Реализованная архитектура
Архитектура реализованной модели представлена ниже.
![photo_2024-06-13_00-52-23](https://github.com/sadevans/EfLipSystem/assets/82286355/97dcf13e-f5d0-48e7-89b5-1869628d7248)

Модель состоит из блока 3D сверточной сети, отмасштабированной EfficientNetV2, энкодера трансформера и блока временной сверточной сети (TCN). Розовым на рисунке обозначена внешняя часть сети (frontend), выполняющая извлечение признакв, оранжевым - внутренняя часть сети (backend), отвечающая за обработку признаков.

Строение ее составляющих блоков предствлено ниже.
![tcn_arch](https://github.com/sadevans/EfLipReading/assets/82286355/d0e26768-1f2a-41a8-92d0-fbad7dabf427)
![mbconv_arch](https://github.com/sadevans/EfLipReading/assets/82286355/2883a0ab-dec3-4c82-8ac9-d07890d4b061)

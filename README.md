# Infrastructure object recognition using satellite data

Команда Darkhole AI

Стек: OpenCV, YOLOv8, OSM

Наше решение испольузет instance segmentation, используя лишь некоторые аугментации из YOLO, также используется дополнительный алгоритм постобработки

## Структура

- datasets - обработанные, разбитые на тайлы, датасеты
- prepare - ряд функций для препроцессинга датасетов
  - convert_yolo.py - функции для преобразования маски в lables и val выборки для yolo
  - download.py - создание датасета(спутинкового изображения и маски) из данных с ArcGIS и OpenStreetMap (Overpass API)
  - parse_vendor.py - разбиение картинок из исходного датасета на датасеты с тайлами
  - split.py - разбиение изображения на тайлы
  - unite.py - собрать несколько датасетов в один для yolo
- satellite
  - download.py - функции для загрузки данных из ArcGIS
  - merge.py - функция для объединения тайлов в картинку
  - overpass.py - функции для получения информации из OpenStreetMap
  - utils.py - вспомогательные функции для работы с координатами
- vendor - данные для тренировок
- analysis.ipynb - Jupyter Notebook с анализом исходного датасета
- bboxes.py - данные для создания собственного датасета
- contours.ipynb - ноутбук с тестами постобработки маски
- data.yaml - конфиг обучения для yolo
- dataset.py - скрипт для создания датасета для YOLO (create_yolo) или для DeepLab(create_semantic). Датасет создаётся на основе своих данных и предоставленных
- deeplab_test.ipynb - ноутбук с обучением deeplab неройсети
- deeplab.py - predict для deeplab
- main.py - основной файл с метриками и интерфейсом
- train_yolo_collab.ipynb - ноутбук с тренировкой yolo в Google Collab
- unet.py - predict для нейросети unet (не используется)
- yolo.py - predict для yolo

## Где взять модель для YOLO и DeepLAB

В разделе Github Releases этого репозитория можно скачать модели

# Readme. Classifier module
---
This module is an image classifier model with simple flask deploy block.
The model defines whether there is a phone on the image or not.
Baseline prototype. Eventual task is more complicated and demands more of data engineering, however ML-part should not be more sophisticated than the prototype.

## Structure
phone-classifier-v2 project nests two packages: classifier and flapi; test_inference directory; standart starting .py files

**Classifier** содержит пакет models, папки notebooks и trained_models а также модули dir_renaming.py, preprocess.py, train.py и inference.py.
- .models хранит классы моделей, наследующие torch.nn.Module. Инстанс модели необходим для загрузки сохраненного состояния - обученной модели с весами, - хранящегося в папке trained_models.
- Notebooks хранит Jupiter ноутбуки, использовавшиеся при тестировании версий.

- Preprocess.py отвечает за предобработку датасета перед обучением. Частично задействован в боевой сборке (см. далее).
- Train.py отвечает за обучение моделей.
- Inference.py отвечает за "боевое" использование обученных моделей. Пока содержит единственную функцию judge_image, определяющую, является ли изображение фотографией     телефона или нет. __*Задействует статические методы из класса PhoneDataset модуля preprocess.*__ В свою очередь используется в пакете flapi для формирования response.

**Flapi** - пакет, реализующий простой flask-сервер с шаблоном формой. POST метод отправляет на сервер изображение и возвращет True/False, после чего на странице дается соответствующий комментарий. Запускается из корня скриптом run_service.py.

**Test_inference** содержит небольшой набор изображений для тестирования функционала.

**Нужное для работы** - flapi, preprocess, inference, папка trained_models. 
Обученные модели весят немало, гит их грузить отказывается. 

---
## TODO. v0.1.3
1. Flask-обертка чрезвычайно проста и неказиста.
2. Требуется переработка кода с целью упрощения и повышения читаемости, для рефакторинга.
3. Сериализация, логи, .env.




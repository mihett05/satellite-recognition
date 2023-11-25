import os
import cv2 as cv
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

from yolo import predict_yolo, yolo


def metrics(predict: np.ndarray, true_name: str):
    true = np.array(cv.imread(true_name, cv.COLOR_BGR2GRAY).flatten())
    true[true > 0] = 1
    print(sklearn.metrics.f1_score(true, predict.flatten() / 255, pos_label=1))


def run():
    print(
        "Выберите модель\n1)YOLOv8n (использовалась для генерации финального результата)\n2)DeepLab"
    )
    model = 0
    while not model:
        inp = input()
        if not inp.isdigit():
            print("[ERROR] Введите число - номер модели")
        elif int(inp) not in [1, 2]:
            print("[ERROR] Введите число от 1 до 2")
        else:
            model = int(inp)
    while True:
        path = input("Введите путь до картинки: ")
        if os.path.exists(path):
            break
        print("[ERROR] Путь не найден")

    if model == 1:
        predict = predict_yolo(yolo, path)
    else:
        from deeplab import predict_deeplab

        predict = predict_deeplab(path)

    cv.imwrite("debug.png", predict)
    cv.imwrite("mask.png", predict / 255)
    print("Результат записан в mask.png")

    img = cv.imread(path)
    img |= np.stack(
        (predict, np.zeros_like(predict), np.zeros_like(predict)), axis=2
    ).astype(dtype=np.uint32)

    plt.imshow(img)
    plt.show()


def do_test():
    for i in range(8):
        predict = predict_yolo(yolo, f"test/images/test_image_{i:03}.png")
        cv.imwrite(f"test/masks/test_mask_{i:03}.png", predict)


if __name__ == "__main__":
    run()

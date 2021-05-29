from pathlib import Path
from multiprocessing.dummy import Pool
import cv2
from rich.console import Console
import numpy as np
from PIL import Image

print = Console().print
base = Path("D:/", "dataset", "yolo")
classes = open(str(base) + "/classes.txt", "r").read().strip().split()
cls2idx = {j: i for i, j in enumerate(classes)}


def process_line(p: str):
    img_p = base / p
    name = img_p.stem
    print(img_p)
    # img = Image.open(str(img_p).strip())
    # img = np.array(img)
    img = cv2.imread(str(img_p).strip())
    H, W, C = img.shape
    print(f"H-->{H}, W-->{W}, C-->{C}")
    lines = open(str(img_p.parents[0]) + "/" + name + ".txt", 'r').readlines()
    for l in lines:
        print(l)
        l = l.split()



        label = int(l[0])
        x = float(l[1])
        y = float(l[2])
        w = float(l[3])
        h = float(l[4])
        # ==============================================
        x1 = int((x - w / 2) * W)
        x2 = int((x + w / 2) * W)
        y1 = int((y - h / 2) * H)
        y2 = int((y + h / 2) * H)

        if x1 < 0:
            x1 = 0
        if x2 > W - 1:
            x2 = W - 1
        if y1 < 0:
            y1 = 0
        if y2 > H - 1:
            y2 = H - 1

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # ====================================================
        # topX = int((x - w / 2) * W)
        # topY = int((y - h / 2) * H)
        # botX = int((x + w / 2) * W)
        # botY = int((y + h / 2) * H)
        # c = np.random.choice(range(255), size=(len(classes), 3))
        # c = [[int(i) for i in k] for k in c]
        # print(topX, topY, botX, botY)
        # img = cv2.rectangle(img, (topX, topY), (botX, botY), tuple(c[cls2idx[label]]), 1, cv2.LINE_AA)
        # img = cv2.putText(img, f"{label}", (topX, topY - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, tuple(c[cls2idx[label]]), 1,
        #                   cv2.LINE_AA)
        img = cv2.putText(img, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 100), 1,
                          cv2.LINE_AA)

    cv2.imwrite(f"D:/dataset/{str(name).strip()}.jpg", img)


lines = open(str(base / "images.txt"), 'r').readlines()
with Pool(64) as p:
    for line in lines:
        p.apply_async(process_line, args=(line,)).get()

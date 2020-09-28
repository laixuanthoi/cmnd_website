from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
import glob

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False

detector = Predictor(config)


# img = "image/20.JPG"
# img = cv2.imread(img)
# pil_img = Image.fromarray(img)
# start = time.time()
# s = detector.predict(pil_img)
# print(s, "predicted: ", time.time()-start)
# cv2.imshow("image", img)
# cv2.waitKey(0)

for f in glob.glob("image/*.JPG"):
    img = cv2.imread(f)
    pil_img = Image.fromarray(img)
    start = time.time()
    s = detector.predict(pil_img)
    print(s, "predicted: ", time.time()-start)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)

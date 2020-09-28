from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from PIL import Image

import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = ""


config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False

detector = Predictor(config)
img = "image/17.JPG"
img = Image.open(img)
# plt.imshow(img)
s = detector.predict(img)
print(s)

import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False

detector = Predictor(config)
img = "C:/Users/MSi/Desktop/9.JPG"
img = Image.open(img)
plt.imshow(img)
s = detector.predict(img)
print(s)

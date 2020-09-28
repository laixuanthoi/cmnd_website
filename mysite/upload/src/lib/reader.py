from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
detector = Predictor(config)


def readText(images):
    texts = []
    for img in images:
        img = Image.fromarray(img)
        t = detector.predict(img)
        texts.append(t)
    return texts

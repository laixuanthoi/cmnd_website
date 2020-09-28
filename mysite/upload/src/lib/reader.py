import pytesseract
import cv2
import numpy as np
import re
pytesseract.pytesseract.tesseract_cmd = "D:/tesseract-ocr/tesseract.exe"


class READER:
    def textClean(self, text):
        vietnamese_text = r'[a-z0-9A-Z_àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]+'
        regex = re.compile(vietnamese_text)
        match = re.findall(regex, text)
        return match

    def readText(self, image):
        config = '--psm 13'
        lang = 'vie_best'
        return pytesseract.image_to_string(image, lang=lang, config=config)

    def readNumber(self, image):
        config = '--psm 13'
        lang = 'vie_best'
        return pytesseract.image_to_string(image, lang=lang, config=config)

    def readMultiText(self, images):
        texts = []
        for img in images:
            text = self.readText(img)
            matchs = self.textClean(text)
            if len(matchs) == 0:
                text = "#"
            else:
                text = ''.join(matchs)
            texts.append(text)
        return texts

    def readMultiNumber(self, images):
        texts = []
        for img in images:
            text = self.readNumber(img)
            matchs = self.textClean(text)
            if len(matchs) == 0:
                text = "#"
            else:
                text = ''.join(matchs)
            texts.append(text)
        return texts

    def readNguyenQuan(self, boxes):
        full_text = ''
        for row in boxes:
            for img in row:
                text = self.readText(img)
                matchs = self.textClean(text)
                if len(matchs) == 0:
                    text = "#"
                else:
                    text = ''.join(matchs)
                full_text += '{} '.format(text)
            full_text += '\n'
        return full_text

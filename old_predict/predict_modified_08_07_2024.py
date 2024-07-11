# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import numpy as np
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


import easyocr

import cv2
from datetime import datetime
import os
import glob
import pytesseract
from google.cloud import vision

# Assuming your JSON key is uploaded to your Colab root and named 'vision_key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision_key.json'

from google.cloud import vision
import keras_ocr

reader = easyocr.Reader(['en'], gpu=True)

# Initialize the Keras OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def enhance_image(img):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # # Apply Gaussian Blur
    # blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Apply adaptive thresholding
    thresh_img = cv2.adaptiveThreshold(gray_img, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

    # # Adjust contrast
    # alpha = 1.5  # Contrast control (1.0-3.0)
    # beta = 0  # Brightness control (0-100)
    # contrast_img = cv2.convertScaleAbs(thresh_img, alpha=alpha, beta=beta)

    return thresh_img

def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve OCR accuracy
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to get a binary image
    binary = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return binary

def perform_ocr_on_image(img, coordinates):
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray_img)
    # print(results)

    text = ""
    for res in results:
        if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
            text = res[1]

    #     # Debugging: Save the enhanced image
    #     # Enhance image before OCR
    # enhanced_img = enhance_image(cropped_img)
    #
    # # # Debugging: Save the enhanced image
    # saved_image = "enhanced_image.png"
    # cv2.imwrite(saved_image, enhanced_img)
    #
    # img = cv2.imread(saved_image, 0)
    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    #
    # result = reader.readtext(blur)
    # for (bbox, text, prob) in result:
    #     print(f'Text: {text}, Probability: {prob}')

    return str(text)


def perform_ocr_on_image2(im, coors):
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im = im[y:h, x:w]
    conf = 0.2

    # gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    preprocessed_img = preprocess_image(im)
    results = reader.readtext(preprocessed_img)
    ocr = ""

    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) > 1 and len(results[1]) > 6 and results[2] > conf:
            ocr = result[1]

    return str(ocr)

def perform_ocr_on_image_google_vision(img, coordinates):
    """
    Extracts text from an image using Google Cloud Vision API.

    Args:
        path: Path to the image file.

    Returns:
        String containing the extracted text.
    """
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]
    path = "../enhanced_image.png"
    cv2.imwrite(path, cropped_img)

    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message))

    return texts[0].description


def perform_ocr_with_keras(img, coordinates):
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]

    # Keras OCR expects images in RGB format
    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    # Perform OCR using Keras OCR pipeline
    prediction_groups = pipeline.recognize([cropped_img_rgb])

    text = ""
    for predictions in prediction_groups:
        for prediction in predictions:
            text += prediction[0] + " "

    return text.strip()

# def perform_ocr_on_image_tesseract(img, coordinates):
#     x, y, w, h = map(int, coordinates)
#     crop = img[y:h, x:w]
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bilateralFilter(gray, 10, 20, 20)
#
#     text = pytesseract.image_to_string(gray).strip()
#     text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '')
#
#     return str(text)

def perform_ocr_on_image_tesseract(img, coordinates):
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # saved_image = f"saved_image_{timestamp}.png"
    processed_images_dir = "../processed_images"
    saved_image = os.path.join(processed_images_dir, f"saved_image_{timestamp}.png")
    cv2.imwrite(saved_image, cropped_img)

    img = cv2.imread(saved_image)

    # Preprocessing the image starts

    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    # Creating a copy of image
    im2 = img.copy()

    # A text file is created and flushed
    file = open("../recognized.txt", "w+")
    file.write("")
    file.close()

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]

        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped, config=r'--oem 3 --psm 6')

    return str(text)
class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                
                
                text_ocr = perform_ocr_on_image(im0,xyxy)
                text_ocr2 = perform_ocr_on_image2(im0, xyxy)
                text_ocr3 = perform_ocr_on_image_google_vision(im0, xyxy)
                # text_ocr4 = perform_ocr_with_keras(im0, xyxy)
                text_ocr5 = perform_ocr_on_image_tesseract(im0, xyxy)
                print("EasyOCR 1: "+text_ocr2)
                print("EasyOCR 2: "+text_ocr2)
                print('Google Vision OCR: '+text_ocr3)
                print('Tesseract: ' + text_ocr5)
                # print('Keras OCR: ' + text_ocr4)
                label = text_ocr
                
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt" #"best.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
#     predictor = DetectionPredictor(cfg)
#     predictor()
# def predict(cfg, model_path="ultralytics/runs/detect/train_model/weights/best.pt", source_path="assets/images/1.jpg"):
#     cfg.model = model_path
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
#     cfg.source = source_path
#     predictor = DetectionPredictor(cfg)
#     predictor()

def predict(cfg, model_path="ultralytics/runs/detect/train_model/weights/best.pt", source_dir="assets/uploaded"):
    cfg.model = model_path
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    image_paths = glob.glob(os.path.join(source_dir, "*.jpeg"))

    for image_path in image_paths:
        cfg.source = image_path
        predictor = DetectionPredictor(cfg)
        predictor()

if __name__ == "__main__":
    predict()

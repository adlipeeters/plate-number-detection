# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import numpy as np
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import easyocr
import pytesseract

import pytesseract
import cv2

import os
import glob

from PIL import Image

reader = easyocr.Reader(['en'], gpu=True)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Deskew the image if necessary
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# get grayscale image
def get_grayscale_operation(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# thresholding
def thresholding_operation(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# opening - erosion followed by dilation
def opening_operation(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny_operation(image):
    return cv2.Canny(image, 100, 200)


def perform_ocr_on_image(img, coordinates):
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]
    # Debugging: Save the enhanced image
    saved_image = "saved_image.png"
    cv2.imwrite(saved_image, cropped_img)

    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray_img)
    print(results)

    try:
        # # Apply adaptive thresholding
        # thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #
        # # Apply dilation and erosion
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)
        # eroded_img = cv2.erode(dilated_img, kernel, iterations=1)
        #
        # # Debugging: Save the preprocessed image
        # preprocessed_image_path = "preprocessed_image.png"
        # cv2.imwrite(preprocessed_image_path, eroded_img)

        # Perform OCR using Tesseract
        custom_config = r'--oem 3 --psm 6'
        # custom_config = r'--oem 1 -l eng --psm 3'
        img = cv2.imread(saved_image)
        gray = get_grayscale_operation(img)
        thresholding = thresholding_operation(gray)
        opening = opening_operation(gray)
        canny = canny_operation(gray)
        # text = pytesseract.image_to_string(eroded_img, config=custom_config)
        text = pytesseract.image_to_string(gray, config=custom_config)
        print('tesseract result')
        print(text)
    except Exception as e:
        print(f"Error with Tesseract OCR: {e}")

    text = ""
    for res in results:
        if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
            text = res[1]

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

                text_ocr = perform_ocr_on_image(im0, xyxy)
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

def predict(cfg, model_path="ultralytics/runs/detect/train_model/weights/best.pt", source_dir="assets/images"):
    cfg.model = model_path
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    image_paths = glob.glob(os.path.join(source_dir, "*.jpg"))

    for image_path in image_paths:
        cfg.source = image_path
        predictor = DetectionPredictor(cfg)
        predictor()


if __name__ == "__main__":
    predict()

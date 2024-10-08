import os
import hydra
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import easyocr
import cv2
from omegaconf import OmegaConf
from google.cloud import vision
import time
import logging

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision_key.json'

reader = easyocr.Reader(['en'], gpu=True)

def perform_ocr_on_image(img, coordinates):
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray_img)
    text = ""
    for res in results:
        if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
            text = res[1]
    return str(text)


# Set up logging
logging.basicConfig(filename='google_vision_logs.log', level=logging.INFO, format='%(asctime)s %(message)s')


def perform_ocr_on_image_google_vision(img, coordinates):
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]
    path = "enhanced_image.png"
    cv2.imwrite(path, cropped_img)

    # Start measuring time
    start_time = time.time()

    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    # Measure the time after the response
    end_time = time.time()
    time_taken = end_time - start_time

    # Log the time taken
    logging.info(f"OCR request took {time_taken:.2f} seconds")

    texts = response.text_annotations
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message))
    print(texts)
    if texts:
        return texts[0].description
    else:
        logging.info("No text detected by Google Vision API.")
        return ""  # Return an empty string if no text was detected
    
def normalize_plate(text_ocr):
    # Remove 'RO' if it appears at the beginning
    if text_ocr.upper().startswith('RO'):
        text_ocr = text_ocr[2:]
    
    # Remove special characters, keeping only letters and numbers
    normalized_plate = ''.join(char for char in text_ocr if char.isalnum())
    
    return normalized_plate

class DetectionPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detected_plates = []  # List to store detected number plates
        self.saved_image_path = None  # Variable to store the path of the saved image

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

                # text_ocr = perform_ocr_on_image(im0, xyxy)
                text_ocr = perform_ocr_on_image_google_vision(im0, xyxy)
                text_ocr = normalize_plate(text_ocr)
                label = text_ocr

                if text_ocr:  # Append detected text to the list
                    self.detected_plates.append(text_ocr)

                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        if self.args.save:  # Save the annotated image
            self.saved_image_path = str(self.save_dir / p.name)  # Store the path to the saved image
            cv2.imwrite(self.saved_image_path, im0)  # Save the image

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    predictor = DetectionPredictor(cfg)
    predictor()
    detected_plates = predictor.detected_plates
    saved_image_path = predictor.saved_image_path
    return detected_plates, saved_image_path

def run_prediction(image_path):
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    cfg.model = "ultralytics/runs/detect/train_model/weights/best.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = image_path
    return predict(cfg)


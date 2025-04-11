import cv2
import numpy as np
import torch
import tempfile
import os
from craft_text_detector import Craft


class Segmenter:
    def crop(self, image, config):
        raise NotImplementedError


class ManualPlainSegmenter(Segmenter):
    def crop(self, image, config):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((config['kHeight'], config['kWidth']), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

        segments = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= config['minHeight']:
                y_start = max(0, y - config['overlapUp'])
                y_end = min(gray.shape[0], y + h + config['overlapUp'])
                segment = image[y_start:y_end, :]
                if np.mean(segment) < 253.5:
                    segments.append(segment)
        return segments


class ManualRuledSegmenter(Segmenter):
    def crop(self, image, config):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1] // 20, 1))
        detect_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(detect_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        line_boundaries = [cv2.boundingRect(c)[1] for c in contours]

        if len(line_boundaries) < 2:
            return []

        segments = []
        for idx in range(len(line_boundaries) - 1):
            y_top = line_boundaries[idx] - config['overlapUp']
            y_bottom = line_boundaries[idx + 1] + config['overlapDn']
            if y_bottom - y_top > config['minHeight']:
                segment = image[y_top:y_bottom, :]
                if config['minWhite'] < np.mean(segment) < config['maxWhite']:
                    segments.append(segment)
        return segments


class AutoPlainSegmenter(Segmenter):
    def __init__(self):
        self.craft = Craft(crop_type="box", cuda=torch.cuda.is_available())

    def crop(self, image, config=None):
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = temp_file.name
        cv2.imwrite(temp_path, image)
        temp_file.close()

        try:
            result = self.craft.detect_text(temp_path)
            boxes = sorted(result["boxes"], key=lambda box: box[0][1])

            segments = []
            for box in boxes:
                x1, y1 = int(box[0][0]), int(box[0][1])
                x2, y2 = int(box[2][0]), int(box[2][1])
                cropped = image[y1:y2, x1:x2]
                if cropped.shape[0] > 10 and cropped.shape[1] > 10:
                    segments.append(cropped)

            return segments

        finally:
            os.remove(temp_path)

    def unload(self):
        self.craft.unload_craftnet_model()
        self.craft.unload_refinenet_model()
        torch.cuda.empty_cache()


def get_segmenter(mode, page_type):
    if mode == "manual" and page_type == "plain":
        return ManualPlainSegmenter()
    elif mode == "manual" and page_type == "ruled":
        return ManualRuledSegmenter()
    elif mode == "automatic" and page_type == "plain":
        return AutoPlainSegmenter()
    else:
        raise ValueError("Unsupported mode/page_type combo")

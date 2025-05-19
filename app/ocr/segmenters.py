import cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt


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


class AutoSegmenter(Segmenter):
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en',
                             det=True, rec=False,
                             det_db_box_thresh=0.2,
                             det_db_unclip_ratio=3.5,
                             det_box_type='poly',
                             use_dilation=True,
                             det_db_score_mode='slow')

    @staticmethod
    def crop_poly(img, box, expand_pixels):
        # Create an empty black mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Draw the polygon on the mask
        pts = np.array([box], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)

        # --- ✨ DILATE the mask to expand the polygon ✨ ---
        kernel_size = expand_pixels * 2 + 1  # Make sure kernel is odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded_mask = cv2.dilate(mask, kernel)

        # Apply mask to the image
        masked = cv2.bitwise_and(img, img, mask=expanded_mask)

        # Create white background
        white_bg = np.ones_like(img, dtype=np.uint8) * 255

        # Invert expanded mask to get outside region
        inverted_mask = cv2.bitwise_not(expanded_mask)

        # Combine masked region + white background
        result = cv2.bitwise_or(masked, cv2.bitwise_and(white_bg, white_bg, mask=inverted_mask))

        # Find new bounding rect
        x, y, w, h = cv2.boundingRect(expanded_mask)

        # Crop
        cropped = result[y:y + h, x:x + w]

        height, width = cropped.shape[:2]

        if height < 384 or width < 384:
            cropped = cv2.resize(cropped, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        return cropped

    @staticmethod
    def sort_polys(polys):
        # Sort by the **topmost point** first, then leftmost
        def get_top_left_point(poly):
            # poly = list of (x, y)
            topmost = min(poly, key=lambda p: (p[1], p[0]))  # first by y, then by x
            return topmost

        # Get top-left for each poly
        poly_top_lefts = [(poly, get_top_left_point(poly)) for poly in polys]

        # Sort first by Y (top to bottom), then by X (left to right)
        poly_top_lefts.sort(key=lambda x: (x[1][1], x[1][0]))

        # Return sorted polys
        sorted_polys = [ptl[0] for ptl in poly_top_lefts]
        return sorted_polys

    def crop(self, image, config=None):

        # Convert OpenCV image (BGR) to PIL (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run OCR
        result = self.ocr.ocr(image, cls=True)
        boxes = [line[0] for line in result[0]]



        boxes = self.sort_polys(boxes)
        segments = []

        for idx, box in enumerate(boxes):
            cropped_img = self.crop_poly(image, box,expand_pixels=3)
            #plt.imshow(cropped_img)
            #plt.show()
            if cropped_img.shape[0] > 10 and cropped_img.shape[1] > 10:
                segments.append(cropped_img)

        return segments

    def unload(self):
        # PaddleOCR doesn’t need unloads like CRAFT did,
        # but we can still clear GPU memory if used.
        import paddle
        if paddle.device.get_device() != 'cpu':
            paddle.device.cuda.empty_cache()


def get_segmenter(mode, page_type):
    if mode == "manual" and page_type == "plain":
        return ManualPlainSegmenter()
    elif mode == "manual" and page_type == "ruled":
        return ManualRuledSegmenter()
    elif mode == "automatic":
        return AutoSegmenter()
    else:
        raise ValueError("Unsupported mode/page_type combo")

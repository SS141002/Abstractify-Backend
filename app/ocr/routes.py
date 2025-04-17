from flask import Blueprint, jsonify, request, abort
from app.socketio_instance import socketio
from .models import trocr_processor, trocr_model, device
from .segmenters import get_segmenter
from app.utils.image_processing import read_image
import easyocr
import json

ocr_bp = Blueprint('ocr', __name__)


def parse_int_field(field_name, default):
    val = request.form.get(field_name, str(default))
    return int(val) if val.strip().isdigit() else default


@ocr_bp.route('/ocr', methods=['POST'])
def handle_ocr():
    files = request.files.getlist("images")
    if not files or len(files) == 0:
        abort(400, description="No images uploaded")

    total_files = len(files)
    current_file = 0
    socketio.emit("ocr_progress", {"current_file": None, "status": "files received", "progress": 0})

    for field in ['ocrMode', 'segmentMode', 'pageType']:
        if field not in request.form:
            abort(400, description=f'no {field} part in request')

    ocr_mode = request.form.get('ocrMode')
    segment_mode = request.form.get('segmentMode')
    page_type = request.form.get('pageType')

    if ocr_mode == "typed":
        if 'languages' not in request.form:
            abort(400, description='No languages provided')
        languages = json.loads(request.form['languages'])
        try:
            results = {}
            reader = easyocr.Reader(lang_list=languages, gpu=True, model_storage_directory="./models/easyocr/")
            for file in files:
                filename = file.filename
                image = read_image(file)
                result = reader.readtext(image=image)
                file_text = ""
                for i in result:
                    file_text += i[1] + " "
                results[filename] = file_text.strip()  # trim trailing space
                current_file += 1
                socketio.emit("ocr_progress",
                              {"current_file": None, "status": f"processing files {current_file}/{total_files}",
                               "progress": int((current_file / total_files) * 100)})
            return jsonify(results), 200
        except Exception as e:
            abort(500, description=str(e))

    else:
        try:
            for field in ['kHeight', 'kWidth', 'overlapUp', 'overlapDn', 'minHeight', 'minWhite', 'maxWhite']:
                if field not in request.form:
                    abort(400, description=f'no {field} part in request')
            config = {
                'kHeight': parse_int_field('kHeight', 1),
                'kWidth': parse_int_field('kWidth', 50),
                'overlapUp': parse_int_field('overlapUp', 10),
                'overlapDn': parse_int_field('overlapDn', 10),
                'minHeight': parse_int_field('minHeight', 20),
                'minWhite': parse_int_field('minWhite', 0),
                'maxWhite': parse_int_field('maxWhite', 255)
            }
            segmenter = get_segmenter(segment_mode, page_type)
            results = {}
            fname = ""
            for file in files:
                filename = file.filename
                fname = filename
                image = read_image(file)
                segments = segmenter.crop(image, config)
                full_text = ""
                for segment in segments:
                    pixel_values = trocr_processor(images=segment, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(device)
                    generated_ids = trocr_model.generate(pixel_values)
                    generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    full_text += generated_text + "\n"
                results[filename] = full_text.strip()
                current_file += 1
                socketio.emit("ocr_progress",
                              {"current_file": None, "status": f"processing files {current_file}/{total_files}",
                               "progress": int((current_file / total_files) * 100)})
            return jsonify(results), 200
        except Exception as e:
            print(e)
            abort(500, description=str(e) + fname)

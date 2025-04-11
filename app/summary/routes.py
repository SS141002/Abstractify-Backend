from flask import Blueprint, jsonify, request, abort
from werkzeug.utils import secure_filename
from .models import *  # Assumes tokenizer, model, and device are imported from here
import os
import io
from docx import Document
import fitz  # PyMuPDF

summary_bp = Blueprint('summary', __name__)


def extract_text_from_txt(file_obj):
    try:
        file_obj.seek(0)
        content = file_obj.read()
        if isinstance(content, bytes):
            return content.decode('utf-8')
        return content
    except Exception as e:
        abort(500, description=f"Error reading TXT file: {str(e)}")


def extract_text_from_docx(file_obj):
    try:
        file_obj.seek(0)
        document = Document(file_obj)
        return "\n".join([para.text for para in document.paragraphs])
    except Exception as e:
        abort(500, description=f"Error reading DOCX file: {str(e)}")


def extract_text_from_pdf(file_obj):
    try:
        file_obj.seek(0)
        pdf_data = file_obj.read()
        pdf_stream = io.BytesIO(pdf_data)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text
    except Exception as e:
        abort(500, description=f"Error reading PDF file: {str(e)}")


def extract_text(file_obj, filename):
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".txt":
        return extract_text_from_txt(file_obj)
    elif ext in [".doc", ".docx"]:
        return extract_text_from_docx(file_obj)
    elif ext == ".pdf":
        return extract_text_from_pdf(file_obj)
    else:
        abort(400, description="Unsupported file type")


@summary_bp.route('/summary', methods=['POST'])
def handle_summary():
    # Priority: If a file is provided, use it; otherwise, use JSON text.
    if 'file' in request.files and request.files['file'].filename:
        file_obj = request.files['file']
        # Sanitize the file name using secure_filename
        filename = secure_filename(file_obj.filename)
        long_text = extract_text(file_obj, filename)
        try:
            minlength = int(request.form.get('min', ''))
            maxlength = int(request.form.get('max', ''))
        except Exception:
            abort(400,
                  description='Minimum and Maximum length parameters must be provided and valid when uploading a file')
    else:
        data = request.get_json()
        if not data:
            abort(400, description="No JSON data provided and no file uploaded")
        if 'min' not in data:
            abort(400, description='Minimum Length not provided')
        if 'max' not in data:
            abort(400, description='Maximum Length not provided')
        if 'text' not in data:
            abort(400, description='Text not provided')
        minlength = data['min']
        maxlength = data['max']
        long_text = data['text']

    if not long_text:
        abort(400, description="No text content extracted from the input")

    try:
        inputs = tokenizer(long_text, max_length=1024, return_tensors="pt", truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=maxlength,
            min_length=minlength,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return jsonify({'summary': summary_text})
    except Exception as e:
        abort(500, description=f"Summarization error: {str(e)}")

from flask import Blueprint, jsonify, request, abort
from .models import *

grammar_bp = Blueprint('grammar', __name__)


@grammar_bp.route('/grammar', methods=['POST'])
def handle_grammar():
    data = request.get_json()

    if 'text' not in data:
        abort(400, description='Text not provided')

    long_text = data['text']

    try:
        # Load the model when needed
        gram_tokenizer, gram_model = get_grammar_model()

        input_text = "Fix grammatical errors in this text: " + long_text
        input_ids = gram_tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        output = gram_model.generate(input_ids, max_length=1024)
        result = gram_tokenizer.decode(output[0], skip_special_tokens=True)

        unload_grammar_model()
        return jsonify({'text': result})

    except Exception as e:
        print(e)
        abort(500, description=str(e))

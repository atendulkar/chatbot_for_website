from flask import Flask, render_template, request, jsonify
from chatbot import build_knowledge_base, generate_response_vector

app = Flask(__name__)
kb_chunks, kb_vectors = build_knowledge_base(max_pages=50, chunk_size=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('message')
    response = generate_response_vector(user_input, kb_chunks, kb_vectors, top_n=2)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

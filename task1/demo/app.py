from flask import Flask, render_template, request
from search_engine import VectorSearchEngine
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Пути к данным
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'tfidf_results', 'lemmas')
INDEX_FILE = os.path.join(BASE_DIR, '..', 'index.txt')

# Инициализация поисковой системы
search_engine = VectorSearchEngine(DATA_DIR, INDEX_FILE)


@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ""

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            results = search_engine.search(query, top_k=10)

    return render_template('index.html', query=query, results=results)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
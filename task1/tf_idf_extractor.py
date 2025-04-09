import os
import math
from collections import defaultdict, Counter

# Конфигурация путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENS_DIR = os.path.join(BASE_DIR, 'lemma_token_output', 'tokens')
LEMMAS_DIR = os.path.join(BASE_DIR, 'lemma_token_output', 'lemmas')
OUTPUT_DIR = os.path.join(BASE_DIR, 'tfidf_results')
TOKENS_OUTPUT = os.path.join(OUTPUT_DIR, 'tokens')
LEMMAS_OUTPUT = os.path.join(OUTPUT_DIR, 'lemmas')

# Создаем выходные директории
os.makedirs(TOKENS_OUTPUT, exist_ok=True)
os.makedirs(LEMMAS_OUTPUT, exist_ok=True)


def load_data():
    """Загружает токены и леммы, строит индексы"""
    token_files = [f for f in os.listdir(TOKENS_DIR) if f.endswith('.txt')]
    lemma_files = [f for f in os.listdir(LEMMAS_DIR) if f.endswith('.txt')]

    # Инвертированные индексы
    token_index = defaultdict(list)
    lemma_index = defaultdict(list)

    # Статистика по документам
    token_stats = defaultdict(dict)
    lemma_stats = defaultdict(dict)

    all_docs = set()

    # Обработка токенов
    for filename in token_files:
        doc_id = filename.replace('.txt', '')
        all_docs.add(doc_id)
        with open(os.path.join(TOKENS_DIR, filename), 'r', encoding='utf-8') as f:
            tokens = f.read().split()
            counter = Counter(tokens)
            for token, count in counter.items():
                token_index[token].append(doc_id)
                token_stats[doc_id][token] = count

    # Обработка лемм
    for filename in lemma_files:
        doc_id = filename.replace('.txt', '')
        all_docs.add(doc_id)
        with open(os.path.join(LEMMAS_DIR, filename), 'r', encoding='utf-8') as f:
            lemma_forms = defaultdict(list)
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                lemma = parts[0]
                forms = parts[1:]
                lemma_index[lemma].append(doc_id)
                lemma_stats[doc_id][lemma] = len(forms)  # Количество словоформ

    return (token_index, lemma_index,
            token_stats, lemma_stats,
            all_docs)


def compute_tfidf():
    # 1. Загрузка данных
    token_index, lemma_index, token_stats, lemma_stats, all_docs = load_data()
    total_docs = len(all_docs)

    # 2. Расчет IDF с защитой от нулей
    token_idf = {term: math.log((total_docs + 1) / (len(set(docs)) + 1))
                 for term, docs in token_index.items()}

    lemma_idf = {lemma: math.log((total_docs + 1) / (len(set(docs)) + 1))
                 for lemma, docs in lemma_index.items()}

    # 3. Обработка каждого документа
    for doc_id in all_docs:
        # Обработка токенов
        if doc_id in token_stats:
            total_tokens = sum(token_stats[doc_id].values())
            with open(os.path.join(TOKENS_OUTPUT, f'{doc_id}.txt'), 'w', encoding='utf-8') as f:
                for token, count in token_stats[doc_id].items():
                    tf = count / total_tokens
                    idf = token_idf.get(token, 0)
                    tfidf = tf * idf
                    if tfidf >= 0.0001:  # Фильтр
                        f.write(f"{token} {idf:.6f} {tfidf:.6f}\n")

        # Обработка лемм
        if doc_id in lemma_stats:
            total_forms = sum(lemma_stats[doc_id].values())
            with open(os.path.join(LEMMAS_OUTPUT, f'{doc_id}.txt'), 'w', encoding='utf-8') as f:
                for lemma, count in lemma_stats[doc_id].items():
                    tf = count / total_forms
                    idf = lemma_idf.get(lemma, 0)
                    tfidf = tf * idf
                    if tfidf >= 0.0001:  # Фильтр
                        f.write(f"{lemma} {idf:.6f} {tfidf:.6f}\n")


if __name__ == "__main__":
    print("Запуск расчета TF-IDF с раздельными папками...")
    compute_tfidf()
    print(f"Результаты сохранены:\n- Токены: {TOKENS_OUTPUT}\n- Леммы: {LEMMAS_OUTPUT}")
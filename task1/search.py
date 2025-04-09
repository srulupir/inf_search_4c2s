import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFIDF_RESULTS_DIR = os.path.join(BASE_DIR, "tfidf_results", "lemmas")


# 1. Загрузка данных из файлов
def load_tfidf_data(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Папка не найдена: {folder_path}")

    all_words = set()
    documents_tfidf = []
    filenames = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".txt"):
            continue

        full_path = os.path.join(folder_path, filename)
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                doc_words = {}
                for line in file:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue

                    word = parts[0].rstrip(':')
                    tf, idf = map(float, parts[1:3])
                    doc_words[word] = tf * idf
                    all_words.add(word)

                documents_tfidf.append(doc_words)
                filenames.append(filename)
        except Exception as e:
            print(f"Ошибка чтения {filename}: {e}")

    vocab = sorted(all_words)
    tfidf_matrix = np.zeros((len(documents_tfidf), len(vocab)))

    for i, doc in enumerate(documents_tfidf):
        for j, word in enumerate(vocab):
            tfidf_matrix[i, j] = doc.get(word, 0.0)

    return tfidf_matrix, vocab, filenames


#2. Обработка запроса
def process_query(query, vocab):
    query_words = query.lower().split()
    vector = np.zeros(len(vocab))

    for word in query_words:
        if word in vocab:
            idx = vocab.index(word)
            vector[idx] += 1

    if np.sum(vector) > 0:
        vector /= len(query_words)

    return vector


# 3. Поиск документов
def find_top_documents(query_vector, tfidf_matrix, filenames, top_k=5):
    similarities = cosine_similarity(query_vector.reshape(1, -1), tfidf_matrix)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(filenames[i], similarities[i]) for i in top_indices]


# --- Основной цикл ---
def main():
    print(f"\nЗагрузка данных из: {TFIDF_RESULTS_DIR}")

    try:
        tfidf_matrix, vocab, filenames = load_tfidf_data(TFIDF_RESULTS_DIR)
        print(f"Успешно загружено:\n- Документов: {len(filenames)}\n- Уникальных слов: {len(vocab)}")

        while True:
            print("\n" + "=" * 50)
            query = input("Поисковый запрос: ").strip()

            if query.lower() == 'exit':
                break
            if not query:
                print("Введите непустой запрос")
                continue

            query_vector = process_query(query, vocab)
            results = find_top_documents(query_vector, tfidf_matrix, filenames)

            print("\nТоп результатов:")
            for i, (filename, score) in enumerate(results, 1):
                print(f"{i}. {filename} (сходство: {score:.4f})")

    except Exception as e:
        print(f"Критическая ошибка: {e}")


if __name__ == "__main__":
    main()
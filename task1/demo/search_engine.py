import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorSearchEngine:
    def __init__(self, data_dir, index_file):
        self.data_dir = data_dir
        self.url_mapping = self._load_url_mapping(index_file)
        self.tfidf_matrix, self.vocab, self.filenames = self._load_data()

    def _load_url_mapping(self, index_file):
        mapping = {}
        with open(index_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    filename = f"page_{parts[0]}.txt"
                    url = parts[1]
                    mapping[filename] = url
        return mapping

    def _load_data(self):
        """Загрузка TF-IDF данных"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Директория не найдена: {self.data_dir}")

        all_words = set()
        documents_tfidf = []
        filenames = []

        for filename in sorted(os.listdir(self.data_dir)):
            if not filename.endswith(".txt"):
                continue

            full_path = os.path.join(self.data_dir, filename)
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

    def search(self, query, top_k=10):
        """Поиск документов"""
        query_vector = self._process_query(query)
        if query_vector is None:
            return []

        similarities = cosine_similarity(
            query_vector.reshape(1, -1),
            self.tfidf_matrix
        )[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [
            {
                'document': self.filenames[idx],
                'url': self.url_mapping.get(self.filenames[idx], "#"),
                'score': float(similarities[idx]),
                'content': self._get_document_preview(idx)
            }
            for idx in top_indices
        ]

    def _process_query(self, query):
        """Обработка поискового запроса"""
        query_words = query.lower().split()
        vector = np.zeros(len(self.vocab))

        for word in query_words:
            if word in self.vocab:
                idx = self.vocab.index(word)
                vector[idx] += 1

        if np.sum(vector) > 0:
            vector /= len(query_words)
            return vector
        return None

    def _get_document_preview(self, doc_idx):
        filename = self.filenames[doc_idx]
        try:
            with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                return f.read()[:300] + "..."
        except:
            return "Не удалось загрузить содержимое"
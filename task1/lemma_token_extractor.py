import os
import re
from bs4 import BeautifulSoup
import nltk
import pymorphy2
from langdetect import detect
from nltk.tokenize import RegexpTokenizer

nltk.download('punkt')
nltk.download('stopwords')

# скачиваем стоп слова
stop_words_ru = set(nltk.corpus.stopwords.words('russian'))
stop_words_en = set(nltk.corpus.stopwords.words('english'))

morph = pymorphy2.MorphAnalyzer()

folder_path = 'pages'
output_folder = 'lemma_token_output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

tokenizer = RegexpTokenizer(r'\w+')

# извлекаем из html
def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        return soup.get_text(separator=' ')


def clean_and_tokenize(text):
    tokens = tokenizer.tokenize(text.lower())
    clean_tokens = []
    for token in tokens:
        # пропускаем токен, если содержит цифры
        if re.search(r'\d', token):
            continue
        # опредеяем язык токена
        try:
            language = detect(token)
            # если русский и английский язык - ок, продолжаем
            if language == 'ru' and token not in stop_words_ru:
                clean_tokens.append(token)
            elif language == 'en' and token not in stop_words_en:
                clean_tokens.append(token)
        except:
            pass  # игнорируем токены, которые ен можем определить
    return clean_tokens


def lemmatize_tokens(tokens):
    lemma_dict = {}
    for token in tokens:
        lemma = morph.parse(token)[0].normal_form
        if lemma not in lemma_dict:
            lemma_dict[lemma] = []
        lemma_dict[lemma].append(token)
    # убираем дубликаты для кадждой леммы
    for lemma in lemma_dict:
        lemma_dict[lemma] = list(set(lemma_dict[lemma]))
    return lemma_dict


def main():
    for filename in os.listdir(folder_path):
        if filename.endswith('.html'):
            file_path = os.path.join(folder_path, filename)
            print(f"Обработка файла: {filename}")

            text = extract_text_from_html(file_path)

            tokens = clean_and_tokenize(text)
            unique_tokens = sorted(set(tokens))

            lemma_dict = lemmatize_tokens(unique_tokens)

            # записываем токены в файл
            tokens_file = os.path.join(output_folder, f'tokens_{filename}.txt')
            with open(tokens_file, 'w', encoding='utf-8') as f:
                for token in unique_tokens:
                    f.write(f'{token}\n')

            # записываем леммы в файл
            lemmas_file = os.path.join(output_folder, f'lemmas_{filename}.txt')
            with open(lemmas_file, 'w', encoding='utf-8') as f:
                for lemma, token_list in sorted(lemma_dict.items()):
                    tokens_str = ' '.join(sorted(token_list))
                    f.write(f'{lemma} {tokens_str}\n')

    print('Файлы сохранены в папку:', output_folder)

if __name__ == '__main__':
    main()

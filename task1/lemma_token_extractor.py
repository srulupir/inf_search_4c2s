import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import nltk
import pymorphy2
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

INPUT_FOLDER = 'pages'
OUTPUT_FOLDER = 'lemma_token_output'
TOKENS_FOLDER = os.path.join(OUTPUT_FOLDER, 'tokens')
LEMMAS_FOLDER = os.path.join(OUTPUT_FOLDER, 'lemmas')
WORKERS = 4

logging.basicConfig(
    filename='processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

morph = pymorphy2.MorphAnalyzer()
stop_words_ru = set(stopwords.words('russian'))
stop_words_en = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\b[а-яА-ЯёЁa-zA-Z]+\b')


def setup_folders():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(TOKENS_FOLDER, exist_ok=True)
    os.makedirs(LEMMAS_FOLDER, exist_ok=True)

#очищаем текст от тэгов
def extract_text_from_html(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')

            for element in soup(['script', 'style', 'meta', 'link',
                                 'footer', 'header', 'nav', 'noscript',
                                 'iframe', 'svg', 'img', 'button']):
                element.decompose()

            article_body = soup.find('div', class_='tm-article-body') or soup.find('article')

            if article_body:
                for pre in article_body.find_all('pre'):
                    pre.decompose()
                for code in article_body.find_all('code'):
                    code.decompose()
                text = article_body.get_text(' ', strip=True)
            else:
                text = soup.get_text(' ', strip=True)

            text = re.sub(r'\s+', ' ', text)
            return text.lower()

    except Exception as e:
        logging.error(f"Ошибка при чтении {file_path}: {str(e)}")
        return ""


def clean_and_tokenize(text):
    try:
        tokens = tokenizer.tokenize(text)

        # фильтр стоп слов и коротких слов
        clean_tokens = [
            token for token in tokens
            if (len(token) > 2 and
                token not in stop_words_ru and
                token not in stop_words_en and
                not token.isdigit())
        ]
        return clean_tokens
    except Exception as e:
        logging.error(f"Ошибка токенизации: {str(e)}")
        return []


def lemmatize_tokens(tokens):
    lemma_dict = {}
    try:
        for token in tokens:
            try:
                parsed = morph.parse(token)[0]
                if parsed.score >= 0.5:
                    lemma = parsed.normal_form
                    if lemma not in lemma_dict:
                        lemma_dict[lemma] = set()
                    lemma_dict[lemma].add(token)
            except:
                continue
    except Exception as e:
        logging.error(f"Ошибка лемматизации: {str(e)}")

    return {lemma: sorted(token_set) for lemma, token_set in lemma_dict.items()}


def save_results(filename, tokens, lemma_dict):
    try:
        base_name = os.path.splitext(filename)[0]

        # сохраняем токены
        tokens_file = os.path.join(TOKENS_FOLDER, f'{base_name}.txt')
        with open(tokens_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(set(tokens))))

        # сохраняем леммы
        lemmas_file = os.path.join(LEMMAS_FOLDER, f'{base_name}.txt')
        with open(lemmas_file, 'w', encoding='utf-8') as f:
            for lemma, token_list in sorted(lemma_dict.items()):
                f.write(f"{lemma}: {', '.join(token_list)}\n")

        logging.info(f"Успешно обработан: {filename}")
    except Exception as e:
        logging.error(f"Ошибка сохранения {filename}: {str(e)}")


def process_file(filename):
    if filename.endswith('.html'):
        file_path = os.path.join(INPUT_FOLDER, filename)
        try:
            text = extract_text_from_html(file_path)
            if text:
                tokens = clean_and_tokenize(text)
                if tokens:
                    lemma_dict = lemmatize_tokens(tokens)
                    save_results(filename, tokens, lemma_dict)
        except Exception as e:
            logging.error(f"Ошибка при обработке {filename}: {str(e)}")


def main():
    setup_folders()
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.html')]

    if not files:
        logging.warning(f"В папке {INPUT_FOLDER} не найдено HTML-файлов")
        return

    logging.info(f"Начало обработки {len(files)} файлов")

    # многопоточная обработка
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        executor.map(process_file, files)

    logging.info("Обработка завершена")
    print(f"Готово! Обработано {len(files)} файлов.")

if __name__ == '__main__':
    main()
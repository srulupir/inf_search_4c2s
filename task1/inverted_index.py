import os
import json
import re
from collections import defaultdict

LEMMAS_FOLDER = "lemma_token_output/lemmas"
INDEX_FOLDER = "inverted_index_output"
INDEX_FILE = os.path.join(INDEX_FOLDER, "inverted_index.json")


def build_index():
    print("[ИНДЕКС] Начало построения индекса...")
    os.makedirs(INDEX_FOLDER, exist_ok=True)

    index = defaultdict(list)
    doc_ids = set()

    lemma_files = [f for f in os.listdir(LEMMAS_FOLDER) if f.endswith('.txt')]
    if not lemma_files:
        print(f"[ОШИБКА] В папке '{LEMMAS_FOLDER}' нет .txt файлов!")
        return None, None

    print(f"[ИНДЕКС] Обработка {len(lemma_files)} файлов с леммами...")

    for filename in lemma_files:
        doc_id = filename[:-4]
        print(f"[ИНДЕКС] Обработка документа: {doc_id}")
        with open(os.path.join(LEMMAS_FOLDER, filename), 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    lemma, tokens = line.split(':', 1)
                    lemma = lemma.strip()
                    index[lemma].append(doc_id)
                    doc_ids.add(doc_id)

    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(dict(index), f, ensure_ascii=False, indent=2)

    print(f"[ИНДЕКС] Успешно построен индекс. Сохранен в: {os.path.abspath(INDEX_FILE)}")
    print(f"[ИНДЕКС] Статистика: {len(doc_ids)} документов, {len(index)} уникальных терминов")

    return index, doc_ids


def search(query, index, doc_ids):
    print(f"\n[ПОИСК] Начало обработки запроса: '{query}'")

    if not index:
        print("[ОШИБКА] Индекс не загружен!")
        return []

    def parse_expression(expr, depth=0):
        indent = "  " * depth
        expr = expr.strip().lower()
        print(f"{indent}[ПАРСЕР] Уровень {depth}: разбираем '{expr}'")

        # Если выражение уже содержит ID документов (после обработки скобок)
        if all(part.startswith('page_') for part in expr.split()):
            print(f"{indent}[DOCIDS] Возвращаем готовые ID документов: {expr.split()}")
            return set(expr.split())

        # Базовый случай - одиночный термин
        if ' ' not in expr and '(' not in expr:
            if expr.startswith('not '):
                term = expr[4:].strip()
                print(f"{indent}[NOT] Поиск документов БЕЗ термина '{term}'")
                result = doc_ids - set(index.get(term, []))
                print(f"{indent}[NOT] Найдено {len(result)} документов без '{term}'")
                return result
            print(f"{indent}[ТЕРМ] Поиск термина '{expr}'")
            result = set(index.get(expr, []))
            print(f"{indent}[ТЕРМ] Найдено {len(result)} документов с '{expr}'")
            return result

        # Обрабатываем вложенные скобки
        while '(' in expr:
            print(f"{indent}[СКОБКИ] Обнаружены скобки в '{expr}'")
            expr = re.sub(
                r'\(([^()]+)\)',
                lambda m: ' '.join(sorted(parse_expression(m.group(1), depth + 1))),
                expr
            )
            print(f"{indent}[СКОБКИ] После обработки: '{expr}'")

        # Разбиваем на OR части
        or_parts = [part.strip() for part in re.split(r'\s+or\s+', expr)]
        if len(or_parts) > 1:
            print(f"{indent}[OR] Разбиваем на {len(or_parts)} частей: {or_parts}")
            result = set()
            for part in or_parts:
                part_result = parse_expression(part, depth + 1)
                print(f"{indent}[OR] Часть '{part}' → {len(part_result)} документов")
                result.update(part_result)
            print(f"{indent}[OR] Итоговый результат: {len(result)} документов")
            return result

        # Разбиваем на AND части
        and_parts = [part.strip() for part in re.split(r'\s+and\s+', expr)]
        if len(and_parts) > 1:
            print(f"{indent}[AND] Разбиваем на {len(and_parts)} частей: {and_parts}")
            result = None
            for part in and_parts:
                part_result = parse_expression(part, depth + 1)
                print(f"{indent}[AND] Часть '{part}' → {len(part_result)} документов")
                if result is None:
                    result = part_result
                else:
                    result &= part_result
                print(f"{indent}[AND] Текущее пересечение: {len(result)} документов")
                if not result:
                    print(f"{indent}[AND] Пустое пересечение, прекращаем обработку")
                    break
            return result or set()

        return set(index.get(expr, []))

    try:
        result = sorted(parse_expression(query))
        print(f"[ПОИСК] Запрос '{query}' обработан. Найдено {len(result)} документов")
        return result
    except Exception as e:
        print(f"[ОШИБКА] Ошибка обработки запроса: {e}")
        return []


def main():
    if not os.path.exists(LEMMAS_FOLDER):
        print(f"[ОШИБКА] Папка '{LEMMAS_FOLDER}' не найдена!")
        return

    print("\nПостроение индекса...")
    index, doc_ids = build_index()
    if not index:
        return

    print("\n" + "=" * 50)
    print("Введите поисковый запрос (AND, OR, NOT, скобки)")
    print("Пример: (python AND django) OR java NOT php")
    print("Введите 'exit' для выхода")
    print("=" * 50)

    while True:
        query = input("\nПоисковый запрос: ").strip()
        if query.lower() == 'exit':
            break

        results = search(query, index, doc_ids)
        print(f"\n[РЕЗУЛЬТАТ] Найдено документов: {len(results)}")
        for doc in results:
            print(f"- {doc}")

    print("\n[СИСТЕМА] Поиск завершен.")


if __name__ == '__main__':
    main()
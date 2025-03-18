import os
import requests
import time
import shutil

#папка для храения файлов
if not os.path.exists('pages'):
    os.makedirs('pages')

def download_page(url, page_num):
    try:
        response = requests.get(url)
        #страница существует
        if response.status_code == 200:
            file_path = f'pages/page_{page_num}.html'
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(response.text)

            #запись номера и ссылки
            with open('index.txt', 'a', encoding='utf-8') as index_file:
                index_file.write(f'{page_num}\t{url}\n')
            print(f'Страница {page_num} успешно скачана: {url}')
            return True
        else:
            print(f'Ошибка при скачивании страницы {page_num}: {url} (Статус: {response.status_code})')
            return False
    except Exception as e:
        print(f'Ошибка при скачивании страницы {page_num}: {url} (Ошибка: {e})')
        return False

base_url = 'https://habr.com/ru/post/'

downloaded_pages = 0

#перебираем страницы от 1 до 100
page_num = 1
while downloaded_pages < 100:
    url = f'{base_url}{page_num}/'  #формируем полный URL с числовым параметром
    if download_page(url, downloaded_pages + 1):
        downloaded_pages += 1
    page_num += 1

    #задержка между запросами 1 с
    time.sleep(1)

print(f'Все) Скачано {downloaded_pages} страниц.')

# Создаем архив, если скачано больше 2MB
if downloaded_pages > 0:
    shutil.make_archive('pages_archive', 'zip', 'pages')

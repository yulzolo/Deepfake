import os

# Путь к папке с файлами
folder_path = r"E:\Deepfake_New\dataset\input_videos"

# Начальное значение счетчика
start_number = 1

# Получаем список всех файлов в папке
files = os.listdir(folder_path)

# Фильтруем только файлы (исключаем папки)
files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

# Сортируем файлы для последовательного переименования
files.sort()

# Счетчик для новых имен
counter = start_number

# Сначала переименовываем все файлы во временные имена
print("Этап 1: Переименование во временные имена...")
temp_counter = 1
for filename in files:
    file_extension = os.path.splitext(filename)[1]
    temp_name = f"temp_{temp_counter:04d}{file_extension}"
    
    old_path = os.path.join(folder_path, filename)
    temp_path = os.path.join(folder_path, temp_name)
    
    os.rename(old_path, temp_path)
    print(f"Временное переименование: {filename} -> {temp_name}")
    temp_counter += 1

# Получаем список временных файлов
temp_files = os.listdir(folder_path)
temp_files = [f for f in temp_files if f.startswith("temp_")]
temp_files.sort()

# Теперь переименовываем временные файлы в нужные имена (v1, v2, v3...)
print(f"\nЭтап 2: Переименование в конечные имена (v{start_number}, v{start_number+1}, v{start_number+2}...)")
counter = start_number
for temp_filename in temp_files:
    file_extension = os.path.splitext(temp_filename)[1]
    new_name = f"v{counter}{file_extension}"
    
    temp_path = os.path.join(folder_path, temp_filename)
    new_path = os.path.join(folder_path, new_name)
    
    os.rename(temp_path, new_path)
    print(f"Конечное переименование: {temp_filename} -> {new_name}")
    counter += 1

print(f"\nГотово! Переименовано файлов с v{start_number} по v{counter-1}")
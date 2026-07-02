import cv2
import numpy as np
import mediapipe as mp

print(f"MediaPipe version: {mp.__version__}")
print(f"MediaPipe attributes: {dir(mp)}")

# Проверяем наличие solutions
if hasattr(mp, 'solutions'):
    print("✓ mp.solutions exists")
    print(f"  solutions attributes: {dir(mp.solutions)}")
else:
    print("✗ mp.solutions does NOT exist")

# Пробуем импортировать face_mesh напрямую
try:
    from mediapipe.python.solutions import face_mesh
    print("✓ Successfully imported face_mesh from mediapipe.python.solutions")
except ImportError as e:
    print(f"✗ Failed to import face_mesh: {e}")

# Создаем тестовое изображение
test_image = np.zeros((256, 256, 3), dtype=np.uint8)
test_image[:] = (255, 255, 255)  # белый фон

# Пробуем использовать face_mesh
try:
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        print(f"Test results: {results.multi_face_landmarks is not None}")
except Exception as e:
    print(f"Error using face_mesh: {e}")

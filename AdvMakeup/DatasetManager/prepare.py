import os
import shutil

INPUT_ROOT = r"./faces"      # folder gốc (500 người)
OUTPUT_ROOT = r"dataset"      # folder output
MAX_IMAGES = 100

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for person in os.listdir(INPUT_ROOT):
        person_path = os.path.join(INPUT_ROOT, person)

        if not os.path.isdir(person_path):
            continue

        # tạo folder output cho mỗi người
        output_person = os.path.join(OUTPUT_ROOT, person)
        os.makedirs(output_person, exist_ok=True)

        # lấy danh sách ảnh theo thứ tự
        images = [
            f for f in os.listdir(person_path)
            if f.lower().endswith(IMAGE_EXT)
        ]

        images.sort()  # quan trọng: theo thứ tự

        selected = images[:MAX_IMAGES]

        for img in selected:
            src = os.path.join(person_path, img)
            dst = os.path.join(output_person, img)

            shutil.copy2(src, dst)

        print(f"{person}: copied {len(selected)} images")


if __name__ == "__main__":
    main()
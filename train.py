import os
import json

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def extract_raw_binary_flow_fast(file_path, threshold=5):
    with open(file_path, "rb") as f:
        data = f.read()

    return [
        1 if abs(data[i] - data[i - 1]) > threshold else 0
        for i in range(1, len(data))
    ]


def load_existing_paths(memory_file):
    if not os.path.exists(memory_file):
        return set()

    existing = set()
    with open(memory_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                existing.add(obj["image_path"])

    return existing


def save_recursive_folder_memory_realtime(
    root_folder,
    label,
    memory_jsonl,
    threshold=5
):
    known_images = load_existing_paths(memory_jsonl)

    added = 0
    skipped = 0

    for root, _, files in os.walk(root_folder):
        for file in files:
            if not file.lower().endswith(IMAGE_EXTS):
                continue

            image_path = os.path.join(root, file)

            if image_path in known_images:
                print(f"⏩ SKIPPED | {image_path}\n")
                skipped += 1
                continue

            print(f"⚡ SCANNING | {label} | {image_path}")

            try:
                binary_flow = extract_raw_binary_flow_fast(
                    image_path, threshold
                )

                record = {
                    "label": label,
                    "image_path": image_path,
                    "binary_flow": binary_flow  # ← horizontal
                }

                # 🔥 FORCE HORIZONTAL BINARY FLOW
                with open(memory_jsonl, "a") as f:
                    json.dump(record, f, separators=(",", ":"))
                    f.write("\n")

                known_images.add(image_path)
                added += 1

                print(f"✅ SAVED | {label} | {image_path}\n")

            except Exception as e:
                print(f"❌ ERROR | {label} | {image_path} | {e}\n")

    print("===== SUMMARY =====")
    print(f"Label   : {label}")
    print(f"Added   : {added}")
    print(f"Skipped : {skipped}")


save_recursive_folder_memory_realtime(
    root_folder="afraz",
    label="afraz.jpg",
    memory_jsonl="memory.jsonl",
    threshold=5
)

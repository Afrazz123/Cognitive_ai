import json
import math
import os

# ---------- CONFIG ----------
THRESHOLD = 5
MEMORY_FILE = "memory.jsonl"
# ----------------------------


# ========= ELECTRIC BINARY FLOW =========
def extract_raw_binary_flow(file_path, threshold=THRESHOLD):
    with open(file_path, "rb") as f:
        data = f.read()

    return [
        1 if abs(data[i] - data[i - 1]) > threshold else 0
        for i in range(1, len(data))
    ]


# ========= LOAD MEMORY (JSONL) =========
def load_memory(memory_file):
    samples = []
    if not os.path.exists(memory_file):
        return samples

    with open(memory_file, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    return samples


# ========= SAVE TO MEMORY =========
def save_to_memory(label, binary_flow):
    record = {
        "label": label,
        "binary_flow": binary_flow
    }
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ========= COSINE SIMILARITY (BINARY) =========
def cosine_similarity_binary(a, b):
    length = min(len(a), len(b))
    if length == 0:
        return 0.0

    dot = 0
    mag_a = 0
    mag_b = 0

    for i in range(length):
        dot += a[i] & b[i]
        mag_a += a[i]
        mag_b += b[i]

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (math.sqrt(mag_a) * math.sqrt(mag_b))


# ========= TEST + SELF CORRECT =========
def test_image_against_memory(image_path):
    memory = load_memory(MEMORY_FILE)

    if not memory:
        print("❌ Memory is empty")
        return

    # Step 1: extract flow
    input_binary = extract_raw_binary_flow(image_path)

    # Step 2: score per label
    label_scores = {}

    for record in memory:
        label = record["label"]
        sim = cosine_similarity_binary(
            input_binary,
            record["binary_flow"]
        )

        if label not in label_scores or sim > label_scores[label]:
            label_scores[label] = sim

    # Step 3: best match
    best_label = max(label_scores, key=label_scores.get)
    best_score = label_scores[best_label]

    # Step 4: report
    print("\n===== TEST RESULT =====")
    print(f"Input image : {image_path}")
    print("\nSimilarity per label:")
    for label, score in sorted(
        label_scores.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {label:<12} → {score:.3f}")

    print("\nFinal decision:")
    print(f"  Predicted as → {best_label}")
    print(f"  Confidence   → {best_score:.3f}")

    # ========= FEEDBACK LOOP =========
    answer = input("\nIs this prediction correct? (y/n): ").strip().lower()

    if answer == "y":
        print("✅ Confirmed. Memory remains unchanged.")
        return

    if answer == "n":
        correct_label = input("Enter correct label: ").strip()
        save_to_memory(correct_label, input_binary)
        print(f"🧠 Memory updated with new understanding → '{correct_label}'")
        return

    print("⚠ Invalid input. Skipping learning.")


# ========= USAGE =========
if __name__ == "__main__":
    test_image_against_memory(
        image_path="write image path"
    )


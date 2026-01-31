import json
import math
import os

# ---------- CONFIG ----------
THRESHOLD = 5
MEMORY_FILE = "memory.jsonl"
# ----------------------------


def extract_raw_binary_flow(file_path, threshold=THRESHOLD):
    with open(file_path, "rb") as f:
        data = f.read()

    return [
        1 if abs(data[i] - data[i - 1]) > threshold else 0
        for i in range(1, len(data))
    ]


def load_memory(memory_file):
    samples = []
    if not os.path.exists(memory_file):
        return samples

    with open(memory_file, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def save_to_memory(label, binary_flow):
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps({
            "label": label,
            "binary_flow": binary_flow
        }) + "\n")


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


def test_image_against_memory(image_path):
    memory = load_memory(MEMORY_FILE)

    if not memory:
        print("❌ Memory is empty")
        return

    input_binary = extract_raw_binary_flow(image_path)

    label_scores = {}
    best_score = -1.0
    best_label = None
    best_match_index = None

    for idx, record in enumerate(memory):
        sim = cosine_similarity_binary(
            input_binary,
            record["binary_flow"]
        )

        label = record["label"]

        if sim > best_score:
            best_score = sim
            best_label = label
            best_match_index = idx

        if label not in label_scores or sim > label_scores[label]:
            label_scores[label] = sim

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

    # ========= FEEDBACK LOOP (FIXED) =========
    answer = input("\nIs this prediction correct? (y/n): ").strip().lower()

    if answer == "y":
        if best_score < 1.0:
            save_to_memory(best_label, input_binary)
            print(
                f"🧠 Reinforced memory → '{best_label}' "
                f"(confidence {best_score:.3f})"
            )
        else:
            print(
                "✅ Perfect confidence (1.0). "
                "No memory update."
            )
        return

    if answer == "n":
        correct_label = input("Enter correct label: ").strip()

        if best_score == 1.0:
            memory[best_match_index]["label"] = correct_label
            with open(MEMORY_FILE, "w") as f:
                for rec in memory:
                    f.write(json.dumps(rec) + "\n")

            print(
                f"🧠 Corrected label → "
                f"'{best_label}' → '{correct_label}'"
            )
        else:
            save_to_memory(correct_label, input_binary)
            print(
                f"🧠 New understanding added → '{correct_label}'"
            )
        return

    print("⚠ Invalid input. Skipping learning.")


if __name__ == "__main__":
    test_image_against_memory("image_path")

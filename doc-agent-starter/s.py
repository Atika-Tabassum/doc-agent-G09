import os
from pathlib import Path
from PIL import Image
import pytesseract

# Explicit search root from your system path
base_dir = Path(
    "/home/suchi/Downloads/DL/doc-agent-template/doc-agent-starter/data"
)

# Target pages and their expected gold line counts
targets = {
    "arogya_p0053": 31,
    "chandalika_p0161": 27,
    "chitrangada_p0130": 31,
    "bishwaparichay_p0343": 16,
}

# Locate matching image files recursively
found_images = {}
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".png"):
            for key in targets:
                if key in file:
                    found_images[key] = Path(root) / file

psm_modes = [3, 4, 6]

print(f"{'Page ID':<22} | {'PSM 3':<8} | {'PSM 4':<8} | {'PSM 6':<8} | {'Gold'}")
print("-" * 65)

for key, gold_count in targets.items():
    img_path = found_images.get(key)

    if not img_path or not img_path.exists():
        print(f"{key:<22} | NOT FOUND")
        continue

    img = Image.open(img_path)
    counts = {}

    for psm in psm_modes:
        data = pytesseract.image_to_data(
            img,
            lang="ben",
            config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
        )

        lines = set()
        for i in range(len(data["text"])):
            if data["text"][i].strip():
                line_id = (
                    data["block_num"][i],
                    data["par_num"][i],
                    data["line_num"][i],
                )
                lines.add(line_id)

        counts[psm] = len(lines)

    print(
        f"{key:<22} | {counts.get(3, 0):<8} | {counts.get(4, 0):<8} | {counts.get(6, 0):<8} | {gold_count}"
    )
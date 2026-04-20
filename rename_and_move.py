import os
import shutil

clean_dir = "clean"
attacked_dir = "attacked"

clean_dir = "training/data/clean_depth"
attack_dir = "training/data/attacked_depth"

os.makedirs(clean_dir, exist_ok=True)
os.makedirs(attack_dir, exist_ok=True)

files = sorted(os.listdir(clean_dir))

for i, file in enumerate(files, start=1):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    new_name = f"car{i}.jpg"

    src_clean = os.path.join(clean_dir, file)
    src_att = os.path.join(attacked_dir, file)

    dst_clean = os.path.join(clean_dir, new_name)
    dst_att = os.path.join(attack_dir, new_name)

    if not os.path.exists(src_att):
        print(f"Skipping {file}, no matching attacked image")
        continue

    shutil.copy(src_clean, dst_clean)
    shutil.copy(src_att, dst_att)

    print(f"Processed {file} → {new_name}")

print("✅ Done!")
import os
import shutil
import ast
import argparse

def main():
    parser = argparse.ArgumentParser(description="Organize ESD dataset into train, val, and test splits")
    parser.add_argument("--esd_dir", type=str, required=True, help="Path to downloaded ESD dataset containing 0011, 0012, etc.")
    parser.add_argument("--output_dir", type=str, required=True, help="Target directory to create train, val, and test folders")
    args = parser.parse_args()

    # 1. Scan downloaded ESD directory recursively to find all wav files
    print("Scanning downloaded ESD dataset...")
    wav_map = {}
    for root, dirs, files in os.walk(args.esd_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_map[file] = os.path.join(root, file)
    print(f"Found {len(wav_map)} wav files in the source ESD directory.")

    # 2. Define the split text files
    code_dir = os.path.dirname(os.path.abspath(__file__))
    splits = {
        "train": os.path.join(code_dir, "train_esd.txt"),
        "val": os.path.join(code_dir, "val_esd.txt"),
        "test": os.path.join(code_dir, "test_esd.txt"),
    }
 
    # 3. Create target directories and copy files
    for split_name, txt_path in splits.items():
        if not os.path.exists(txt_path):
            print(f"Warning: {txt_path} not found. Skipping {split_name} split.")
            continue

        target_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(target_dir, exist_ok=True)
        print(f"Organizing {split_name} split into {target_dir}...")

        copied_count = 0
        missing_count = 0

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = ast.literal_eval(line)
                    # Extract the filename from the audio path in the dict
                    audio_path = data["audio"]
                    filename = os.path.basename(audio_path)
                    
                    if filename in wav_map:
                        src_path = wav_map[filename]
                        dest_path = os.path.join(target_dir, filename)
                        shutil.copy2(src_path, dest_path)
                        copied_count += 1
                    else:
                        missing_count += 1
                except Exception as e:
                    print(f"Error parsing line: {line}. Error: {e}")

        print(f"Copied {copied_count} files to {target_dir}. Missing files: {missing_count}")

if __name__ == "__main__":
    main()

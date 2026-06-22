import os
import shutil

def custom_symlink(src, dst, target_is_directory=False):
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

os.symlink = custom_symlink

from speechbrain.pretrained import EncoderClassifier
import torchaudio
import numpy as np
from tqdm import tqdm

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
folder = "D:/ZEST/ZEST/code/data/test"
target_folder = "D:/ZEST/ZEST/code/x-vectors"
os.makedirs(target_folder, exist_ok=True)
wav_files = os.listdir(folder)
wav_files = [x for x in wav_files if ".wav" in x]
wav_files = [x for x in wav_files if ".npy" not in x]

for i, wav_file in enumerate(tqdm(wav_files)):
    sig, sr = torchaudio.load(os.path.join(folder, wav_file))
    embeddings = classifier.encode_batch(sig)[0, 0, :]
    target_file = os.path.join(target_folder, wav_file.replace(".wav", ".npy"))
    np.save(target_file, embeddings.cpu().detach().numpy())
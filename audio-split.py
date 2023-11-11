from typing import List

from pydub import AudioSegment
from pydub.silence import split_on_silence
from pathlib import Path
import glob


def read_audio_info(audio_info: str) -> list[str]:
    with open(audio_info, 'r') as f:
        data = f.read().split(" ")
    return data


def check_dir_and_get_count(dir: str) -> int:
    Path(dir).mkdir(parents=True, exist_ok=True)
    return len(glob.glob(dir))


if __name__ == '__main__':
    file = "audio/0284b483-76fb-4412-a769-2fc6901f7d5b"
    info = read_audio_info(str(file) + ".txt")
    print(info)
    sound = AudioSegment.from_wav(str(file) + ".wav")
    chunks = split_on_silence(sound, min_silence_len=100, silence_thresh=-40)
    print(chunks)
    for i, chunk in enumerate(chunks):
        count = check_dir_and_get_count("audio/chunks/{0}".format(info[i]))
        chunk.export("audio/chunks/{0}/{1}.wav".format(info[i], count), format="wav")

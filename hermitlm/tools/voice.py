from gtts import gTTS
from io import BytesIO

try:
    # pydub imports the stdlib `audioop` module at import time, which was
    # removed in Python 3.13 (PEP 594). Guard the import so a missing/broken
    # pydub only disables speed-up instead of crashing anything that imports
    # this module (e.g. the whole Discord bot).
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

def text_to_mp3(text, speed=1.25):
    mp3_fp = BytesIO()

    text = text.replace("\n", " ")[:300]

    # Generate base audio
    tts = gTTS(text=text, lang="en", tld="co.uk")
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    if AudioSegment is None:
        return mp3_fp

    try:
        audio = AudioSegment.from_file(mp3_fp, format="mp3")
    except FileNotFoundError:
        # pydub needs ffmpeg/ffprobe for MP3 processing. If it is not installed,
        # return the normal gTTS MP3 so Discord audio still works.
        mp3_fp.seek(0)
        return mp3_fp

    # Speed up
    faster = audio._spawn(
        audio.raw_data,
        overrides={
            "frame_rate": int(audio.frame_rate * speed)
        }
    ).set_frame_rate(audio.frame_rate)

    out_fp = BytesIO()
    faster.export(out_fp, format="mp3")
    out_fp.seek(0)

    return out_fp

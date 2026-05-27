from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment

def text_to_mp3(text, speed=1.25):
    mp3_fp = BytesIO()

    text = text.replace("\n", " ")[:300]

    # Generate base audio
    tts = gTTS(text=text, lang="en", tld="co.uk")
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

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

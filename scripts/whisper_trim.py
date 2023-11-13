import os
import whisper
import pandas as pd

SAMPLE_RATE = 16000


def main():
    conv_name = "NY798_1015_Part1_conversation1"

    model_name = "large-v2"
    audio_filename = f"/projects/HASSON/247/data/conversations-car/798/{conv_name}/audio/{conv_name}_deid.wav"
    result_filename = f"{conv_name}.csv"
    out_filename = f"{conv_name}_trimmed.csv"

    df = pd.read_csv(os.path.join("results", model_name, result_filename))
    audio = whisper.load_audio(audio_filename)

    def check_audio(start, end):
        if sum(audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]) == 0:
            return False
        else:
            return True

    df["silence"] = df.apply(lambda x: check_audio(x.start, x.end), axis=1)
    df = df[df.silence]
    df.drop(columns={"silence"}, inplace=True)
    df.to_csv(os.path.join("results", model_name, out_filename), index=False)

    return


if __name__ == "__main__":
    main()

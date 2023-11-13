import os
import glob
import argparse
import whisper
import pandas as pd

from datetime import datetime
from utils import main_timer

SAMPLE_RATE = 16000


def arg_parser():
    """Argument Parser

    Args:

    Returns:
        args (namespace): commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--conv-idx", type=str, required=True)
    parser.add_argument("--sid", type=str, required=True)
    parser.add_argument("--time-stamp", action="store_true")

    args = parser.parse_args()

    if args.conv_idx.isdigit():  # conv idx
        conv_dir = f"/projects/HASSON/247/data/conversations-car/{args.sid}/*"
        conv_list = sorted(glob.glob(conv_dir))
        conv_name = os.path.basename(conv_list[int(args.conv_idx)])
        args.audio_filename = f"/projects/HASSON/247/data/conversations-car/{args.sid}/{conv_name}/audio/{conv_name}_deid.wav"
    else:  # conv name for testing
        conv_name = args.conv_idx
        args.audio_filename = f"data/tfs/{args.conv_idx}.wav"  # short test

    result_dir = os.path.join("results", args.sid, args.model)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if args.time_stamp:
        args.out_filename = os.path.join(result_dir, f"{conv_name}.csv")
    else:
        args.out_filename = os.path.join(result_dir, f"{conv_name}.txt")
    args.device = "cuda"

    return args


def transcribe(args, filename):
    print(f"Transcribing (Time stamp {args.time_stamp})")
    start_time = datetime.now()
    model = whisper.load_model(args.model)
    result = model.transcribe(filename, word_timestamps=args.time_stamp, language="en")

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")
    return result


def get_datum(args, result):
    print("Getting Transcript")
    start_time = datetime.now()

    data = []
    if args.time_stamp:
        word_idx = 0
        for segment in result["segments"]:
            for word in segment["words"]:
                data.append(pd.DataFrame(word, index=[word_idx]))
                word_idx += 1
        df = pd.concat(data)
    else:
        for segment in result["segments"]:
            data.append(segment["text"])
        df = "".join(data)

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")
    return df


def trim_datum(df, audiofile):
    print("Trimming Datum")
    start_time = datetime.now()
    audio = whisper.load_audio(audiofile)

    def check_audio(start, end):  # check if audio is silent
        if sum(audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]) == 0:
            return False
        else:
            return True

    df["silence"] = df.apply(lambda x: check_audio(x.start, x.end), axis=1)
    df = df[df.silence]  # filter out silence
    df.drop(columns={"silence"}, inplace=True)

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")
    return df


@main_timer
def main():
    # conv_name = "798_1hr"
    # conv_name = "798_30s"
    # conv_name = "NY798_1015_Part1_conversation1"
    # model_name = "large"

    args = arg_parser()

    # transcribe
    result = transcribe(args, args.audio_filename)
    # df = trim_datum(df, audio_filename)
    df = get_datum(args, result)

    # saving results
    if args.time_stamp:  # write datum
        df.to_csv(args.out_filename, index=False)
    else:  # write transcript
        with open(os.path.join(result_dir, args.out_filename), "w") as text_file:
            text_file.write(df)

    return


if __name__ == "__main__":
    main()

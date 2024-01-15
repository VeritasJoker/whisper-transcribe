import argparse
import os
import glob
import shutil
import pandas as pd
from datetime import datetime

import whisperx
import gc


def arg_parser():
    """Argument Parser

    Args:

    Returns:
        args (namespace): commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sid", type=str, required=True)

    args = parser.parse_args()

    result_dir = os.path.join("results/paper-transcripts")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    args.result_dir = result_dir
    args.data_dir = "data/transcripts"

    args.device = "cuda"

    return args


def transcribe(args, audio):
    print("Transcribe with original whisper (batched)")
    start_time = datetime.now()
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)

    model = whisperx.load_model(args.model, args.device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")
    return result


def align(args, audio, result):
    print("Align whisper output")
    start_time = datetime.now()

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=args.device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        args.device,
        return_char_alignments=False,
    )

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")

    return result


def diarization(args, wav, result):
    print("Assign Speaker Labels")
    start_time = datetime.now()

    HF_TOKEN = "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"
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=HF_TOKEN, device=args.device
    )
    diarize_segments = diarize_model(wav)
    # diarize_model(args.audio_filename, min_speakers=2, max_speakers=3)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")
    return result


def get_datum(result):
    print("Getting Datum")
    start_time = datetime.now()

    data = []
    word_idx = 0
    for segment in result["segments"]:
        for word in segment["words"]:
            data.append(pd.DataFrame(word, index=[word_idx]))
            word_idx += 1
    df = pd.concat(data)

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")
    return df


def clean_files(args):
    cleaned_path = os.path.join(args.data_dir, "cleaned")
    if not os.path.exists(cleaned_path):
        os.makedirs(cleaned_path)
    else:
        return

    file_types = {"gt": "*.txt", "human": "*.txt", "wav": "*.wav"}
    for file_type in file_types:
        for file in glob.glob(f"{args.data_dir}/{file_type}/{file_types[file_type]}"):
            print(file)
            base_file = os.path.basename(file)
            base_file = (
                base_file.replace("-", "_")
                .replace("_cropped", "")
                .replace("_datum", "")
            )
            if file_type == "gt":
                base_file = base_file.replace(".txt", "_gt.txt")
            elif file_type == "human":
                base_file = base_file.replace(".txt", "_hu.txt")
            new_file = os.path.join(cleaned_path, base_file)
            shutil.copyfile(file, new_file)

    return


def filter_availability(args):
    # filter if not filtered
    filtered_path = os.path.join(args.data_dir, "filtered")
    if not os.path.exists(filtered_path):
        os.makedirs(filtered_path)
    else:
        return
    # get all wav files
    wavs = glob.glob(f"{args.data_dir}/cleaned/*.wav")
    gt_wavs = []
    both_wavs = []

    for wav in wavs:
        # check if ground truth transcript is available
        gt_trans = wav.replace(".wav", "_gt.txt")
        if os.path.isfile(gt_trans):
            gt_wavs.append(wav)
            human_trans = gt_trans.replace("gt", "hu")
            # check if human transcript is available
            if os.path.isfile(human_trans):
                both_wavs.append(wav)
                # move all the files to filtered folder
                shutil.move(wav, wav.replace("cleaned", "filtered"))
                shutil.move(gt_trans, gt_trans.replace("cleaned", "filtered"))
                shutil.move(human_trans, human_trans.replace("cleaned", "filtered"))

    print(f"Total number of wav files: {len(wavs)}")
    print(f"Number of wav files with ground truth: {len(gt_wavs)}")
    print(f"Number of wav files with both: {len(both_wavs)}")

    return gt_wavs, both_wavs


def main():
    args = arg_parser()

    ####################
    ###### STEP 1 ###### clean file names
    ####################
    clean_files(args)

    ####################
    ###### STEP 2 ###### filter missing files
    ####################
    filter_availability(args)
    # 676: 284 total wav, 283 after filter
    # 717: 414 total wav, 156 after filter

    ####################
    ###### STEP 3 ###### transcription
    ####################
    wavs = glob.glob(f"{args.data_dir}/filtered/NY{args.sid}*.wav")
    for wav in wavs:
        # load audio
        audio = whisperx.load_audio(wav)
        result = transcribe(args, audio)
        result = align(args, audio, result)
        result = diarization(args, wav, result)

        # saving results
        df = get_datum(result)
        out_filename = wav.replace(".wav", "_wx.csv")
        df.to_csv(out_filename, index=False)

    return


if __name__ == "__main__":
    main()

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import sem
from datetime import datetime


SAMPLE_RATE = 512


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

    args = parser.parse_args()

    if args.conv_idx.isdigit():  # conv idx
        conv_dir = f"/projects/HASSON/247/data/conversations-car/{args.sid}/*"
        conv_list = sorted(glob.glob(conv_dir))
        # if args.sid == "7170":  # HACK
        #     conv_list = [conv for conv in conv_list if "txt" not in conv]
        args.conv_name = os.path.basename(conv_list[int(args.conv_idx)])
        args.audio_filename = f"/projects/HASSON/247/data/conversations-car/{args.sid}/{args.conv_name}/audio/{args.conv_name}_envelope.mat"
    else:  # conv name for testing
        args.conv_name = args.conv_idx
        args.audio_filename = f"data/tfs/{args.conv_name}.wav"  # short test

    transcript_dir = os.path.join("results", args.sid, f"{args.model}-x")
    args.transcript_filename = os.path.join(
        transcript_dir, f"{args.conv_name}.csv"
    )

    args.save_dir = os.path.join(
        "results", args.sid, f"{args.model}-x-audio-env-erp"
    )
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


def main():
    args = arg_parser()

    print(args.conv_idx)

    # load audio
    audio_env = loadmat(args.audio_filename)["clinical_freq_audio_envelope"]
    df = pd.read_csv(args.transcript_filename)
    print(f"Loading transcript with {len(df)} rows")

    df = df[~df.start.isna()]
    print(f"Filtering transcript to {len(df)} rows")

    df["onset"] = df.start * SAMPLE_RATE
    df = df.loc[df.onset >= 2 * SAMPLE_RATE]  # trim front
    df = df.loc[df.onset + 2 * SAMPLE_RATE <= len(audio_env)]  # trim end
    print(f"Trimming transcript to {len(df)} rows")

    def get_audio_env_erp(start):
        erp_start = int(start - 2 * SAMPLE_RATE)
        erp_end = int(start + 2 * SAMPLE_RATE + 1)
        audio_erp = audio_env[erp_start:erp_end]
        return np.reshape(audio_erp, audio_erp.shape[0])

    erp = df.apply(lambda x: get_audio_env_erp(x.onset), axis=1)
    erp = np.array(erp.tolist())
    mean_erp = np.mean(erp, axis=0)
    std_erp = sem(erp)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_vals = np.arange(-2, 2.0001, step=1 / 512)
    ax.plot(x_vals, mean_erp, color="red")
    ax.fill_between(
        x_vals, mean_erp - std_erp, mean_erp + std_erp, color="red", alpha=0.2
    )
    ax.axvline(0, ls="dashed", alpha=0.3, c="k")
    ax.set(xlabel="Lag (s)", ylabel="Audio Env", title="Audio Env ERP")

    plt.savefig(os.path.join(args.save_dir, f"{args.conv_name}.jpeg"))

    return


if __name__ == "__main__":
    main()

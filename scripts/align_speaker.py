import argparse
import os
import glob
import pandas as pd
from datetime import datetime


def get_df_pred(predict_file):
    # get prediction
    df_pred = pd.read_csv(predict_file)
    df_pred = df_pred[(~df_pred.start.isna()) & (~df_pred.end.isna())]
    df_pred["onset"] = df_pred.start * 512 - 3000
    df_pred["offset"] = df_pred.end * 512 - 3000
    df_pred["word"] = df_pred["word"].str.strip()

    return df_pred


def get_df(datum_file):
    # original datum
    df = pd.read_csv(
        datum_file,
        sep=" ",
        header=None,
        names=["word", "onset", "offset", "accuracy", "speaker"],
    )
    exclude_words = ["sp", "{lg}", "{ns}", "{LG}", "{NS}", "SP"]
    df = df[~df.word.isin(exclude_words)]

    return df


def get_chunks(df):
    df["utt"] = df.speaker.ne(df.speaker.shift()).cumsum()

    comp = []
    prod = []

    def get_utt_boundary(df):
        bounds = (df["onset"].min(), df["offset"].max())
        if bounds[1] < bounds[0]:
            print("Error in utterance boundary")
            return
        if "Speaker1" in df.speaker.unique():
            prod.append(bounds)
        else:
            comp.append(bounds)
        return

    df.groupby("utt").apply(get_utt_boundary)

    return (comp, prod)


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

    # args.result_dir = os.path.join(
    #     "results", args.sid, f"{args.model}-speaker-aligned"
    # )
    # if not os.path.exists(args.result_dir):
    #     os.makedirs(args.result_dir)

    return args


def main():
    args = arg_parser()

    conv_dir = f"/projects/HASSON/247/data/conversations-car/{args.sid}/*"
    conv_files = sorted(glob.glob(conv_dir))
    convs = [os.path.basename(conv_file) for conv_file in conv_files]

    for conv in convs:
        print(f"Running {conv}")
        if "conversation" not in conv:
            continue
        conv_file = os.path.join(
            f"data/tfs/{args.sid}", f"{conv}_datum_trimmed.txt"
        )
        if args.sid == "625":
            conv_file = os.path.join(
                f"data/tfs/{args.sid}", f"{conv}_datum_conversation_trimmed.txt"
            )
        df = get_df(conv_file)
        predict_file = os.path.join(
            "results", args.sid, args.model, f"{conv}.csv"
        )
        df_pred = get_df_pred(predict_file)

        comp_bounds, prod_bounds = get_chunks(df)

        def align_utt_boundary(onset, offset):
            for prod_bound in prod_bounds:
                if onset > prod_bound[1]:
                    continue
                elif onset >= prod_bound[0] and offset <= prod_bound[1]:
                    return "Speaker1"
                elif offset <= prod_bound[0]:  # early stop
                    break
            for comp_bound in comp_bounds:
                if onset > comp_bound[1]:
                    continue
                elif onset >= comp_bound[0] and offset <= comp_bound[1]:
                    return "Speaker2"
                elif offset <= comp_bound[0]:  # early stop
                    break
            return "None"

        df_pred["speaker"] = df_pred.apply(
            lambda x: align_utt_boundary(x.onset, x.offset), axis=1
        )
        df_pred = df_pred.loc[
            :, ("word", "onset", "offset", "score", "speaker")
        ]

        # outfile = os.path.join(
        #     args.result_dir, f"{conv}-datum_{args.model}.txt"
        # )
        outfile = os.path.join(
            f"/projects/HASSON/247/data/conversations-car/{args.sid}",
            conv,
            "misc",
            f"{conv}_datum_{args.model}.txt",
        )
        df_pred.to_csv(outfile, header=None, index=None, sep=" ", mode="w")

    return


if __name__ == "__main__":
    main()

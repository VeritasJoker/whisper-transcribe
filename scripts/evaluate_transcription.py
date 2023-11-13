import os
import glob
import whisper
import evaluate
import pandas as pd


def evaluate_preds(transcript, transcript_pred):
    # evaluate
    metric = evaluate.load("wer")
    wer = metric.compute(predictions=transcript_pred, references=transcript)
    metric = evaluate.load("cer")
    cer = metric.compute(predictions=transcript_pred, references=transcript)
    print(f"WER: {wer}, CER: {cer}")
    return


def get_pred(predict_file):
    # get prediction
    df_pred = pd.read_csv(predict_file)
    df_pred["onset"] = df_pred.start * 512
    df_pred["offset"] = df_pred.end * 512
    df_pred["word"] = df_pred["word"].str.strip()
    transcript_pred = " ".join(df_pred.word.astype(str).tolist())

    return transcript_pred


def get_transcript(datum_file):
    # original datum
    df = pd.read_csv(
        datum_file,
        sep=" ",
        header=None,
        names=["word", "onset", "offset", "accuracy", "speaker"],
    )
    df["word"] = df["word"].str.strip()
    transcript = " ".join(df.word.astype(str).tolist())

    return transcript


def main():
    models = ["large", "large-v2"]
    models = ["large-v1-x", "large-v2-x", "medium.en-x"]

    conv_dir = "/projects/HASSON/247/data/conversations-car/798/*"
    conv_files = sorted(glob.glob(conv_dir))
    convs = [os.path.basename(conv_file) for conv_file in conv_files]

    for model in models:
        print(f"Running {model}")
        transcripts = []
        preds = []
        for conv in convs:
            print(f"\tRunning {conv}")
            conv_file = os.path.join("data/tfs/798", f"{conv}_datum_trimmed.txt")
            transcript = get_transcript(conv_file)
            predict_file = os.path.join("results", model, f"{conv}.csv")
            pred = get_pred(predict_file)
            evaluate_preds([transcript], [pred])
            transcripts.append(transcript)
            preds.append(pred)
        print("Total:")
        evaluate_preds(transcripts, preds)

    return


if __name__ == "__main__":
    main()

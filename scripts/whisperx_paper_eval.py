import os
import glob
import re
import evaluate
import pandas as pd
import numpy as np

# import whisper

EXCLUDE_WORDS = ["sp", "{lg}", "{ns}", "{LG}", "{NS}", "SP", "{inaudible}"]
NON_WORDS = ["hm", "huh", "mhm", "mm", "oh", "uh", "uhuh", "um"]


def evaluate_preds(transcript, transcript_pred):
    # evaluate
    metric = evaluate.load("wer")
    wer = metric.compute(predictions=transcript_pred, references=transcript)
    metric = evaluate.load("cer")
    cer = metric.compute(predictions=transcript_pred, references=transcript)
    print(f"WER: {wer}, CER: {cer}")
    return (wer, cer)


def evaluate_preds_chunk(grtr, pred):
    pred_len = max(pred.start.max(), pred.end.max())
    bins = np.arange(0, pred_len, step=300)
    bins = np.append(bins, pred_len)
    grtr["chunk"] = pd.cut(grtr.start, bins)
    pred["chunk"] = pd.cut(pred.start, bins)
    grtr_grouped = grtr.groupby("chunk")
    pred_grouped = pred.groupby("chunk")

    results = []
    for (chunk1, grtr_group), (chunk2, pred_group) in zip(grtr_grouped, pred_grouped):
        result = []

        result.append(chunk1)
        result.append(len(grtr_group))
        result.append(len(pred_group))
        result.append(len(grtr_group.speaker.unique()))
        result.append(len(pred_group.speaker.unique()))
        result.append(  # utterance num in gt
            grtr_group.speaker.ne(grtr_group.speaker.shift()).cumsum().max()
        )
        result.append(  # utterance num in pred
            pred_group.speaker.ne(pred_group.speaker.shift()).cumsum().max()
        )
        grtr_trans = " ".join(grtr_group.word.astype(str).tolist())
        pred_trans = " ".join(pred_group.word.astype(str).tolist())
        if len(grtr_group) == 0 or len(pred_group) == 0:  # silence
            continue
        wer, cer = evaluate_preds([grtr_trans], [pred_trans])
        # if wer >= 1.5:
        #     breakpoint()
        result.append(wer)

        results.append(result)

    return results


def evaluate_all_preds(conv, grtr, hu_pred, wx_pred):
    hu_trans, hu_speaker, hu_utt = hu_pred

    result = []
    result.append(conv)
    # length
    result.append(len(grtr))
    result.append(len(hu_trans.split()))
    result.append(len(wx_pred))
    # Unique speakers
    result.append(len(grtr.speaker.unique()))
    result.append(hu_speaker)
    result.append(len(wx_pred.speaker.unique()))
    # Utterance num
    result.append(  # utterance num in gt
        grtr.speaker.ne(grtr.speaker.shift()).cumsum().max()
    )
    result.append(hu_utt)  # utterance num in hu
    result.append(  # utterance num in pred
        wx_pred.speaker.ne(wx_pred.speaker.shift()).cumsum().max()
    )
    # WER
    grtr_trans = " ".join(grtr.word.astype(str).tolist())
    pred_trans = " ".join(wx_pred.word.astype(str).tolist())
    hu_wer, _ = evaluate_preds([grtr_trans], [hu_trans])
    wx_wer, _ = evaluate_preds([grtr_trans], [pred_trans])
    huwx_wer, _ = evaluate_preds([pred_trans], [hu_trans])
    result.append(hu_wer)
    result.append(wx_wer)
    result.append(huwx_wer)

    return result


def get_wx_pred(predict_file):
    # get prediction
    df_pred = pd.read_csv(predict_file)
    df_pred["onset"] = df_pred.start * 512 - 3000
    df_pred["offset"] = df_pred.end * 512 - 3000
    df_pred["word"] = df_pred["word"].str.strip()
    df_pred["word"] = df_pred["word"].apply(  # getting rid of punctuations
        lambda x: str(x).translate(
            str.maketrans("", "", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
        )
    )

    return df_pred


def get_hu_pred(predict_file):
    file = open(predict_file, "r")
    pred = file.readlines()[0]

    # replace some quotes and double quotes
    pred = re.sub("[\(\[].*?[\)\]]", "", pred)
    pred = pred.replace("(", "").replace(")", "")

    # exclude tags and non words
    for word in EXCLUDE_WORDS + NON_WORDS:
        pred = pred.replace(word, "")

    # get speaker strings
    speakers = re.findall(r"Speaker [0-9]+:", pred)
    utt_num = len(speakers)
    speaker_num = len(set(speakers))

    # remove speaker strings
    for speaker in set(speakers):
        pred = pred.replace(speaker, "")

    # remove punctuations
    pred = pred.translate(str.maketrans("", "", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'))

    return pred, speaker_num, utt_num


def get_transcript(datum_file):
    # original datum
    df = pd.read_csv(
        datum_file,
        sep=",",
        header=None,
        names=["word", "onset", "offset", "accuracy", "speaker"],
    )
    df["word"] = df["word"].str.strip()
    df["start"] = (df.onset + 3000) / 512
    df["end"] = (df.offset + 3000) / 512
    df = df[~df.word.isin(EXCLUDE_WORDS)]
    df = df[~df.word.isin(NON_WORDS)]

    return df


def get_speakers_utts(pred):
    pred["utt"] = pred.speaker.ne(pred.speaker.shift()).cumsum()

    def get_speaker_utts(groupdf):
        return len(groupdf.utt.unique())

    speakers_utts = pred.groupby(pred.speaker).apply(get_speaker_utts).tolist()
    return speakers_utts


def main():
    conv_dir = f"data/transcripts/filtered"
    conv_files = sorted(glob.glob(os.path.join(conv_dir, "*wx.csv")))
    results = []
    problem_conv = {}
    for conv in conv_files:
        conv_name = os.path.basename(conv).replace("_wx.csv", "")
        print(f"\tRunning {conv_name}")
        gt_file = conv.replace("wx.csv", "gt.txt")
        hu_file = conv.replace("wx.csv", "hu.txt")

        transcript = get_transcript(gt_file)  # get groundtruth
        wx_pred = get_wx_pred(conv)  # get whisperx trans
        try:
            hu_pred = get_hu_pred(hu_file)  # get human trans
        except:
            problem_conv[conv_name] = "hu_pred can't open"
            continue
        # check if any of 3 is empty
        if len(transcript) == 0:
            problem_conv[conv_name] = "grtr empty"
            continue
        if len(hu_pred[0]) == 0:
            problem_conv[conv_name] = "hu_pred empty"
            continue
        if len(wx_pred) == 0:
            problem_conv[conv_name] = "wx_pred empty"
            continue

        result = evaluate_all_preds(conv_name, transcript, hu_pred, wx_pred)
        results.append(result)

    results = pd.DataFrame(results)
    results.columns = [
        "conv",
        "gt_word_num",
        "hu_word_num",
        "wx_word_num",
        "gt_speaker",
        "hu_speaker",
        "wx_speaker",
        "gt_utt_num",
        "hu_utt_num",
        "wx_utt_num",
        "hu_wer",
        "wx_wer",
        "huwx_wer",
    ]
    results.to_csv(f"paper_results.csv", index=False)

    return


if __name__ == "__main__":
    main()

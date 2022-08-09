# Goals for this tool:
#
# Take in a CSV file with three columns (file_path, transcript),
#    and return information on:
#     1) definitely bad data
#     2) probably bad data
#     3) probably good data

from coqui_stt_training.util.audio import (
    read_audio,
    read_duration,
    get_loadable_audio_type_from_extension,
)
from pandarallel import pandarallel
from pathlib import Path
import pandas as pd
import os
import stt
import librosa
import numpy as np
from tqdm import tqdm


class SttTranscriber:
    """Audio transcriber using coqui stt python client.
        Faster than aws transcriber if you have a gpu so only use with gpu.

    Args:
        model_path (str): Path to tflite file.

        scorer_path (str): Path to language model for scoring.
    """

    def __init__(self, model_path, scorer_path):
        self.model = stt.Model(model_path)
        self.model.enableExternalScorer(scorer_path)

    def transcribe(self, audio_path):
        """Run stt model on audio file and get transcript

        Args:
            audio_path (str): Path to audio file to run stt on.
        """
        data, sr = librosa.load(audio_path, sr=16000)
        wav_norm = data * (32767 / max(0.01, np.max(np.abs(data))))
        return self.model.stt(wav_norm.astype(np.int16))


def get_abspath(df, csv_file):
    def find_abspath(csv_dir, audio_path):
        if os.path.isfile(os.path.abspath(audio_path)):
            return os.path.abspath(audio_path)
        elif os.path.isfile(os.path.abspath(os.path.join(csv_dir, audio_path))):
            return os.path.abspath(os.path.join(csv_dir, audio_path))
        else:
            print("🚨 ERROR: could not resolve abspath for {}".format(audio_path))

    csv_dir = Path(csv_file).parent.resolve().absolute()
    df["abspath"] = df.parallel_apply(
        lambda x: find_abspath(csv_dir, x.wav_filename), axis=1
    )
    return df


def is_audio_readable(df, csv_file, AUDIO_TYPE):
    def is_audio_readable_(AUDIO_TYPE, audio_path):
        try:
            read_audio(AUDIO_TYPE, audio_path)
            return True
        except Exception as exception:
            print(
                " · Cannot read {}, raised exception {}".format(audio_path, exception),
            )
            return False

    print(" · Checking if audio is readable...")

    df["is_readable"] = df.abspath.parallel_apply(lambda x: is_audio_readable_(AUDIO_TYPE, x))
    # df["is_readable"] = df.abspath.parallel_apply(is_audio_readable_)
    df_unreadable = df[df.is_readable == False]
    if df_unreadable.shape[0]:
        print("👀 ─ Found {} unreadable audiofiles".format(df_unreadable.shape[0]))
        csv_name = (
                str(Path(csv_file).resolve().absolute().with_suffix("")) + ".UNREADABLE"
        )
        df_unreadable.to_csv(csv_name, index=False)
        print(" · Wrote unreadable data to {}".format(csv_name))
    else:
        print("😊 Found no unreadable audiofiles")
    df = df[df.is_readable == True]
    return df


def get_audio_type(df):
    # TODO -- check all filenames, not just first
    if not type(df["wav_filename"][0]) is str:
        print("🚨 ERROR: expected string, found type {}".format(type(df["wav_filesize"][0])))
        exit(1)
    AUDIO_TYPE = get_loadable_audio_type_from_extension(os.path.splitext(df["wav_filename"][0])[1].lower())
    if AUDIO_TYPE:
        print(" · First audio file found: {} of type {}".format((df["wav_filename"][0]), AUDIO_TYPE))
        return AUDIO_TYPE
    else:
        print("🚨 ERROR: unknown Audio type file extension")
        exit(1)


def get_num_feat_vectors(df):
    # seconds -> milliseconds, divide by 20 millisecond feature_win_step
    # round up to nearest int
    def calculate_num_feat_vecs(seconds):
        return int(seconds * 1000 / 20)

    print(" · Get num feature vectors...")
    df["num_feat_vectors"] = df.audio_len.parallel_apply(calculate_num_feat_vecs)


def get_audio_duration(df, AUDIO_TYPE):
    # get number of seconds of audio
    def _read_duration(audio):
        read_duration(AUDIO_TYPE, audio)

    print(" · Reading audio duration...")
    df["audio_len"] = df.abspath.parallel_apply(lambda x: read_duration(AUDIO_TYPE, x))
    # df["audio_len"] = df.abspath.parallel_apply(_read_duration)


def get_transcript_length(df):
    print(" · Get transcript length...")
    df["transcript_len"] = df.transcript.parallel_apply(lambda x: len(str(x)))


def remove_offending_input_output_ratio(df, csv_file):
    # CTC algorithm assumes the input is not shorter than the ouput
    # if this is not the case, training breaks, and there's probably
    # something funky with the data
    print(" · Get ratio (num_feats / transcript_len)...")
    df["input_output_ratio"] = df.parallel_apply(
        lambda x: float(x.num_feat_vectors) / float(x.transcript_len), axis=1
    )
    offending_samples_df = df[df["input_output_ratio"] <= 1.0]
    if offending_samples_df.shape[0]:
        print(
            "👀 ┬ Found {} <transcript,clip> pairs with more text than audio (bad for CTC)".format(
                offending_samples_df.shape[0]
            )
        )
        total_hours = (offending_samples_df["audio_len"].sum() / 3600)
        print(
            "   ├ Removing a total of {:0.2f} hours of data from BEST dataset".format(
                total_hours
            )
        )
        csv_name = (
                str(Path(csv_file).resolve().absolute().with_suffix("")) + ".OFFENDING_DATA"
        )
        offending_samples_df.to_csv(csv_name, index=False)
        print("   └ Wrote offending data to {}".format(csv_name))
        df = df[df["input_output_ratio"] > 1.0]
        return df
    else:
        print("😊 Found no offending <transcript,clip> pairs")
        return df


def remove_text_outliers(df, csv_file, num_std_devs, stt_model_path, stt_scorer_path):
    print(" . Running stt model...")
    stt_model = SttTranscriber(stt_model_path, stt_scorer_path)
    stt_texts = []
    for i in tqdm(range(len(df))):
        stt_texts.append(stt_model.transcribe(df.iloc[i]['abspath']))

    df['stt_transcript'] = stt_texts
    #  df['stt_transcript'] = df.abspath.parallel_apply(lambda x: stt_model.transcribe(x), axis=1)
    df['stt_len'] = df.parallel_apply(lambda x: len(x.stt_transcript), axis=1)
    df["text_ratio"] = df.parallel_apply(
        lambda x: float(x.transcript_len) / float(x.stt_len), axis=1
    )
    mean = df["text_ratio"].mean()
    std = df["text_ratio"].std()

    df["text_ratio_deviation"] = df.parallel_apply(
        lambda x: abs(x.text_ratio - mean) - (num_std_devs * std), axis=1
    )
    offending_samples_df = df[df["text_ratio_deviation"] > 0]
    if offending_samples_df.shape[0]:
        print(
            "👀 ┬ Found {} <transcript,stt_text> pairs more than {} standard deviations from the mean".format(
                offending_samples_df.shape[0],
                num_std_devs
            )
        )
        total_hours = (offending_samples_df["audio_len"].sum() / 3600)
        print(
            "   ├ Removing a total of {:0.2f} hours of data from BEST dataset".format(
                total_hours
            )
        )
        csv_name = (
                str(Path(csv_file).resolve().absolute().with_suffix("")) + ".NON_NORMAL"
        )
        offending_samples_df.to_csv(csv_name, index=False)
        print("   └ Wrote offending data to {}".format(csv_name))
        df = df[df["text_ratio_deviation"] <= 0]
        return df
    else:
        print("😊 Found no <transcript,stt_transcript> pairs more than {} standard deviations from the mean".format(num_std_devs))
        return df


def remove_outliers(df, csv_file, num_std_devs):
    # remove all data whose audio_len/trans_len ratio
    # is more than num_std_devs standard deviations from the mean
    print(" · Calculating ratio (num_feats : transcript_len)...")
    df["lens_ratio"] = df.parallel_apply(
        lambda x: float(x.audio_len) / float(x.transcript_len), axis=1
    )
    mean = df["lens_ratio"].mean()
    std = df["lens_ratio"].std()

    df["lens_ratio_deviation"] = df.parallel_apply(
        lambda x: abs(x.lens_ratio - mean) - (num_std_devs * std), axis=1
    )
    offending_samples_df = df[df["lens_ratio_deviation"] > 0]
    if offending_samples_df.shape[0]:
        print(
            "👀 ┬ Found {} <transcript,clip> pairs more than {} standard deviations from the mean".format(
                offending_samples_df.shape[0],
                num_std_devs
            )
        )
        total_hours = (offending_samples_df["audio_len"].sum() / 3600)
        print(
            "   ├ Removing a total of {:0.2f} hours of data from BEST dataset".format(
                total_hours
            )
        )
        csv_name = (
                str(Path(csv_file).resolve().absolute().with_suffix("")) + ".NON_NORMAL"
        )
        offending_samples_df.to_csv(csv_name, index=False)
        print("   └ Wrote offending data to {}".format(csv_name))
        df = df[df["lens_ratio_deviation"] <= 0]
        return df
    else:
        print("😊 Found no <transcript,clip> pairs more than {} standard deviations from the mean".format(num_std_devs))
        return df


def cut_off_audio_len(df, csv_file, max_len):
    # remove all data whose over a max audio len
    offending_samples_df = df[df["audio_len"] > max_len]
    if offending_samples_df.shape[0]:
        print(
            "👀 ┬ Found {} audio clips over {} seconds long".format(
                offending_samples_df.shape[0], max_len
            )
        )
        total_hours = (offending_samples_df["audio_len"].sum() / 3600)
        print(
            "   ├ Removing a total of {:0.2f} hours of data from BEST dataset".format(
                total_hours
            )
        )
        csv_name = (
                str(Path(csv_file).resolve().absolute().with_suffix("")) + ".TOO_LONG"
        )
        offending_samples_df.to_csv(csv_name, index=False)
        print("   └ Wrote too long data to {}".format(csv_name))
        df = df[df["audio_len"] < 30]
        return df
    else:
        print("😊 Found no audio clips over {} seconds in length".format(max_len))
        return df


def cut_off_transcript_len(df, csv_file, min_len):
    # remove all data with transcripts under min length
    offending_samples_df = df[df["transcript_len"] < min_len]
    if offending_samples_df.shape[0]:
        print(
            "👀 ┬ Found {} transcripts under {} characters long".format(
                offending_samples_df.shape[0], min_len
            )
        )
        total_hours = (offending_samples_df["audio_len"].sum() / 3600)
        print(
            "   ├ Removing a total of {:0.2f} hours of data from BEST dataset".format(
                total_hours
            )
        )
        csv_name = (
                str(Path(csv_file).resolve().absolute().with_suffix("")) + ".TOO_SHORT_TRANS"
        )
        offending_samples_df.to_csv(csv_name, index=False)
        print("   └ Wrote too short transcript data to {}".format(csv_name))
        df = df[df["transcript_len"] > 10]
        return df
    else:
        print("😊 Found no transcripts under {} characters in length".format(min_len))
        return df


if __name__ == "__main__":
    import sys

    os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

    csv_file = sys.argv[1]
    num_std_devs = float(sys.argv[2])
    stt_model_path = sys.argv[3]
    stt_scorer_path = sys.argv[4]

    # can't use progress_bar=True https://github.com/nalepae/pandarallel/issues/131
    # in Docker, big CSVs run out of space in /dev/shm https://github.com/nalepae/pandarallel/issues/127
    pandarallel.initialize(use_memory_fs=False)

    ### Must-run ###
    df = pd.read_csv(csv_file)
    if ("transcript" not in df.columns) or ("wav_filename" not in df.columns):
        print("🚨 ERROR: missing headers 'transcript' and 'wav_filename'")
        exit(1)
    df = get_abspath(df, csv_file)
    org_total_samples = df.shape[0]
    print("👀 ─ Found {} <transcript,clip> pairs in {}".format(
        org_total_samples, csv_file
    ))
    AUDIO_TYPE = get_audio_type(df)
    df = is_audio_readable(df, csv_file, AUDIO_TYPE)

    ### Following checks are as you wish ###
    get_audio_duration(df, AUDIO_TYPE)
    org_total_hours = (df["audio_len"].sum() / 3600)
    print(
        "👀 ─ Found a total of {:0.2f} hours of readable data".format(
            org_total_hours
        )
    )

    get_transcript_length(df)
    get_num_feat_vectors(df)

    df = cut_off_audio_len(df, csv_file, 30)
    df = cut_off_transcript_len(df, csv_file, 10)
    df = remove_offending_input_output_ratio(df, csv_file)
    df = remove_outliers(df, csv_file, num_std_devs=num_std_devs)
    df = remove_text_outliers(df, None, num_std_devs=num_std_devs, stt_model_path=stt_model_path, stt_scorer_path=stt_scorer_path)

    csv_name = (
            str(Path(csv_file).resolve().absolute().with_suffix("")) + ".BEST"
    )
    df.to_csv(csv_name, index=False)
    new_total_hours = (df["audio_len"].sum() / 3600)
    total_hours_removed = org_total_hours - new_total_hours
    percent_hours_removed = (total_hours_removed / org_total_hours) * 100
    new_total_samples = df.shape[0]
    total_samples_removed = org_total_samples - new_total_samples
    percent_samples_removed = (total_samples_removed / org_total_samples) * 100
    print(
        "🎉 ┬ Saved a total of {:0.2f} hours of data to BEST dataset".format(
            new_total_hours
        )
    )
    print(
        "   ├ Removed a total of {:0.2f} hours ({:0.2f}% of original data)".format(
            total_hours_removed,
            percent_hours_removed
        )
    )
    print(
        "   ├ Removed a total of {} samples ({:0.2f}% of original data)".format(
            total_samples_removed,
            percent_samples_removed
        )
    )
    print("   └ Wrote best data to {}".format(csv_name))

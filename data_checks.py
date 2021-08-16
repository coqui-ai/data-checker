# goals for this script/ module
# 1) take in a CSV file with three columns (name,size,transcript),
#    and return problematic data (output longer than input)

from coqui_stt_training.util.audio import (
    read_ogg_opus,
    read_ogg_opus_duration,
    read_wav,
    read_wav_duration,
)
from pandarallel import pandarallel
from pathlib import Path
import pandas as pd
import os


def is_audio_readable(df, csv_file):
    def is_wav_readable(audio_path):
        try:
            read_wav(audio_path)
            return True
        except:
            return False

    def is_ogg_opus_readable(audio_path):
        try:
            read_ogg_opus(audio_path)
            return True
        except:
            return False

    if df["abspath"][0].endswith("wav"):
        print("I: Found WAV files...")
        print("I: Checking if WAV readable...")
        df["is_readable"] = df.abspath.parallel_apply(is_wav_readable)
    elif df["abspath"][0].endswith("opus"):
        print("I: Found OPUS files...")
        print("I: Checking if OPUS readable...")
        df["is_readable"] = df.abspath.parallel_apply(is_ogg_opus_readable)
    df_unreadable = df[df.is_readable == False]
    if df_unreadable.shape[0]:
        print("I: Found {} unreadable audiofiles".format(df_unreadable.shape[0]))
        csv_name = (
            str(Path(csv_file).resolve().absolute().with_suffix("")) + ".UNREADABLE"
        )
        df_unreadable.to_csv(csv_name, index=False)
        print("I: Wrote unreadable data to {}".format(csv_name))
    else:
        print("I: Found no unreadable audiofiles")
    df = df[df.is_readable == True]
    return df


def check_audiotype(df):
    # TODO -- check all filenames, not just first
    print("I: first audiofile found: {}".format(df["wav_filename"][0]))

    if not "transcript" in df.columns and "wav_filename" in df.columns:
        print("ERROR: missing headers 'transcript' and 'wav_filename'")
        exit(1)
    if not type(df["wav_filename"][0]) is str:
        print("ERROR: path to file is not string".format(df["wav_filesize"][0]))
        exit(1)
    elif not (
        df["wav_filename"][0].endswith("wav") or df["wav_filename"][0].endswith("opus")
    ):
        print("ERROR: not found '.wav' or '.opus' file extension")
        exit(1)


def get_abspath(csv_dir, audio_path):
    if os.path.isfile(os.path.abspath(audio_path)):
        return os.path.abspath(audio_path)
    elif os.path.isfile(os.path.abspath(os.path.join(csv_dir, audio_path))):
        return os.path.abspath(os.path.join(csv_dir, audio_path))
    else:
        print("ERROR: could not resolve abspath for: ", audio_path)


def get_num_feat_vectors(seconds):
    # seconds -> milliseconds, divide by 20 millisecond feature_win_step
    # round up to nearest int
    return int(seconds * 1000 / 20)


def get_audio_duration(df):
    if df["abspath"][0].endswith("wav"):
        print("I: Found WAV files...")
        print("I: Reading WAV duration...")
        df["audio_len"] = df.abspath.parallel_apply(read_wav_duration)
    elif df["abspath"][0].endswith("opus"):
        print("I: Found OPUS files...")
        print("I: Reading OPUS duration...")
        df["audio_len"] = df.abspath.parallel_apply(read_ogg_opus_duration)


def check_for_offending_transcript_len(csv_file):
    # can't use progress_bar=True https://github.com/nalepae/pandarallel/issues/131
    # in Docker, big CSVs run out of space in /dev/shm https://github.com/nalepae/pandarallel/issues/127
    pandarallel.initialize(use_memory_fs=False)

    # abspath to dir in which CSV file lives
    csv_dir = Path(csv_file).parent.resolve().absolute()
    df = pd.read_csv(csv_file)

    check_audiotype(df)

    df["abspath"] = df.parallel_apply(
        lambda x: get_abspath(csv_dir, x.wav_filename), axis=1
    )

    print("I: Found {} audiofiles".format(df.shape[0]))
    df = is_audio_readable(df, csv_file)
    print("I: Found {} audiofiles".format(df.shape[0]))
    get_audio_duration(df)

    print("I: Get num feature vectors...")
    df["num_feat_vectors"] = df.audio_len.parallel_apply(get_num_feat_vectors)
    print("I: Get transcript length...")
    df["transcript_len"] = df.transcript.parallel_apply(len)
    print("I: Get ratio (num_feats / transcript_len)...")
    df["input_output_ratio"] = df.parallel_apply(
        lambda x: float(x.num_feat_vectors) / float(x.transcript_len), axis=1
    )

    df = df.sort_values(by=["input_output_ratio"])
    offending_samples_df = df[df["input_output_ratio"] <= 1.0]
    if offending_samples_df.shape[0]:
        print(
            "I: Found {} offending <transcript,clip> pairs".format(
                offending_samples_df.shape[0]
            )
        )
        csv_name = (
            str(Path(csv_file).resolve().absolute().with_suffix("")) + ".OFFENDING_DATA"
        )
        offending_samples_df.to_csv(csv_name, index=False)
        print("I: Wrote offending data to {}".format(csv_name))
    else:
        print("I: Found no offending <transcript,clip> pairs")


if __name__ == "__main__":
    import sys

    os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

    CSV = sys.argv[1]
    check_for_offending_transcript_len(CSV)

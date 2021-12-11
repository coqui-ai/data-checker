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


def get_abspath(df, csv_file):
    def find_abspath(csv_dir, audio_path):
        if os.path.isfile(os.path.abspath(audio_path)):
            return os.path.abspath(audio_path)
        elif os.path.isfile(os.path.abspath(os.path.join(csv_dir, audio_path))):
            return os.path.abspath(os.path.join(csv_dir, audio_path))
        else:
            print("ERROR: could not resolve abspath for {}".format(audio_path))
    csv_dir = Path(csv_file).parent.resolve().absolute()
    df["abspath"] = df.parallel_apply(
        lambda x: find_abspath(csv_dir, x.wav_filename), axis=1
    )
    print("I: Found {} <transcript,clip> pairs in {}".format(df.shape[0],csv_file))

def is_audio_readable(df, csv_file, audiotype):
    def is_wav_readable(audio_path):
        try:
            read_wav(audio_path)
            return True
        except Exception as exception:
            print("I: Cannot read {}, raised exception {}".format(audio_path, type(exception).__name__))
            return False

    def is_ogg_opus_readable(audio_path):
        try:
            read_ogg_opus(audio_path)
            return True
        except Exception:
            print("I: Cannot read {}, raised exception {}".format(audio_path, type(exception).__name__))
            return False

    if audiotype == "wav":
        print("I: Checking if WAV readable...")
        df["is_readable"] = df.abspath.parallel_apply(is_wav_readable)
    elif audiotype == "opus":
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


def get_audiotype(df):
    # TODO -- check all filenames, not just first
    if not "transcript" in df.columns and "wav_filename" in df.columns:
        print("ERROR: missing headers 'transcript' and 'wav_filename'")
        exit(1)
    elif not type(df["wav_filename"][0]) is str:
        print("ERROR: expected string, found type {}".format(type(df["wav_filesize"][0])))
        exit(1)
    elif df["wav_filename"][0].endswith("wav") or df["wav_filename"][0].endswith("WAV"):
        print("I: First WAV file found: {}".format(df["wav_filename"][0]))
        return "wav"
    elif df["wav_filename"][0].endswith("opus") or df["wav_filename"][0].endswith("OPUS"):
        print("I: First OPUS file found: {}".format(df["wav_filename"][0]))
        return "opus"
    else:
        print("ERROR: not found WAV or OPUS file extension")
        exit(1)


def get_num_feat_vectors(df):
    # seconds -> milliseconds, divide by 20 millisecond feature_win_step
    # round up to nearest int
    def calculate_num_feat_vecs(seconds):
        return int(seconds * 1000 / 20)
    print("I: Get num feature vectors...")
    df["num_feat_vectors"] = df.audio_len.parallel_apply(calculate_num_feat_vecs)


def get_audio_duration(df, audiotype):
    # get number of seconds of audio
    if audiotype == "wav":
        print("I: Reading WAV duration...")
        df["audio_len"] = df.abspath.parallel_apply(read_wav_duration)
    elif audiotype == "opus":
        print("I: Reading OPUS duration...")
        df["audio_len"] = df.abspath.parallel_apply(read_ogg_opus_duration)
    else:
        print("ERROR: unknown audiotype")
        exit(1)


def get_transcript_length(df):
    print("I: Get transcript length...")
    df["transcript_len"] = df.transcript.parallel_apply(len)

def check_for_offending_input_output_ratio(df, csv_file):
    # CTC algorithm assumes the input is not shorter than the ouput
    # if this is not the case, training breaks, and there's probably
    # something funky with the data
    print("I: Get ratio (num_feats / transcript_len)...")
    df["input_output_ratio"] = df.parallel_apply(
        lambda x: float(x.num_feat_vectors) / float(x.transcript_len), axis=1
    )
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

def get_normal_lengths_ratio(df, csv_file):
    # remove all data whose audio_len/trans_len ratio
    # is more than two standard deviations from the mean
    print("I: Get ratio (num_feats / transcript_len)...")
    df["lens_ratio"] = df.parallel_apply(
        lambda x: float(x.audio_len) / float(x.transcript_len), axis=1
    )
    mean = df["lens_ratio"].mean()
    std = df["lens_ratio"].std()

    df["lens_ratio_deviation"] = df.parallel_apply(
        lambda x: abs(x.lens_ratio - mean) - (2 * std), axis=1
    )
    offending_samples_df = df[df["lens_ratio_deviation"] > 0]
    if offending_samples_df.shape[0]:
        print(
            "I: Found {} non-normal <transcript,clip> pairs".format(
                offending_samples_df.shape[0]
            )
        )
        csv_name = (
            str(Path(csv_file).resolve().absolute().with_suffix("")) + ".NON_NORMAL"
        )
        offending_samples_df.to_csv(csv_name, index=False)
        print("I: Wrote offending data to {}".format(csv_name))
    else:
        print("I: Found no non-normal <transcript,clip> pairs")

def cut_off_audio_len(df, csv_file, max_len):
    # remove all data whose over a max audio len
    offending_samples_df = df[df["audio_len"] > max_len]
    if offending_samples_df.shape[0]:
        print(
            "I: Found {} audio clips over {} seconds long".format(
                offending_samples_df.shape[0], max_len
            )
        )
        csv_name = (
            str(Path(csv_file).resolve().absolute().with_suffix("")) + ".TOO_LONG"
        )
        offending_samples_df.to_csv(csv_name, index=False)
        print("I: Wrote too long data to {}".format(csv_name))
    else:
        print("I: Found no audio clips over {} seconds in length".format(
            max_len
        )
    )

if __name__ == "__main__":
    import sys

    os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

    csv_file = sys.argv[1]

    # can't use progress_bar=True https://github.com/nalepae/pandarallel/issues/131
    # in Docker, big CSVs run out of space in /dev/shm https://github.com/nalepae/pandarallel/issues/127
    pandarallel.initialize(use_memory_fs=False)

    # abspath to dir in which CSV file lives
    df = pd.read_csv(csv_file)
    get_abspath(df, csv_file)
    audiotype = get_audiotype(df)
    df = is_audio_readable(df, csv_file, audiotype)
    get_audio_duration(df, audiotype)
    cut_off_audio_len(df, csv_file, 30)
    df = df[df["audio_len"] < 30]
    get_num_feat_vectors(df)
    get_transcript_length(df)
    check_for_offending_input_output_ratio(df, csv_file)
    get_normal_lengths_ratio(df, csv_file)

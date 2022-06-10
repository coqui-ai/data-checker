# 🫠 data-checker

Code for checking goodness of data for STT and TTS.

### Install with Docker

```
$ git clone https://github.com/coqui-ai/data-checker.git
$ cd data-checker
$ docker build . -t data-checker
```

### Check your install

```
$ docker run data-checker python data_checks.py "/code/data/smoke_test/russian_sample_data/ru.csv" 2
.
.
.
👀 ─ Found 1 <transcript,clip> pairs in /code/data/smoke_test/russian_sample_data/ru.csv
 · First audio file found: ru.wav of type audio/wav
 · Checking if audio is readable...
😊 Found no unreadable audiofiles
 · Reading audio duration...
👀 ─ Found a total of 0.00 hours of readable data
 · Get transcript length...
 · Get num feature vectors...
😊 Found no audio clips over 30 seconds in length
😊 Found no transcripts under 10 characters in length
 · Get ratio (num_feats / transcript_len)...
😊 Found no offending <transcript,clip> pairs
 · Calculating ratio (num_feats : transcript_len)...
😊 Found no <transcript,clip> pairs more than 2.0 standard deviations from the mean
🎉 ┬ Saved a total of 0.00 hours of data to BEST dataset
   ├ Removed a total of 0.00 hours (0.00% of original data)
   ├ Removed a total of 0 samples (0.00% of original data)
   └ Wrote best data to /code/data/smoke_test/russian_sample_data/ru.BEST
```

### Run on your data

`data-checker` assumes your CSV has two columns: `wav_filename` and `transcript`. Note that you don't actually need to use WAV files, but the header still should be `wav_filename`.

```
$ docker run data-checker --mount "type=bind,src=/path/to/my/local/data,dst=/mnt" python data_checks.py "/mnt/my-data.csv" 2
```

# 馃珷 data-checker

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
馃憖 鈹? Found 1 <transcript,clip> pairs in /code/data/smoke_test/russian_sample_data/ru.csv
 路 First audio file found: ru.wav of type audio/wav
 路 Checking if audio is readable...
馃槉 Found no unreadable audiofiles
 路 Reading audio duration...
馃憖 鈹? Found a total of 0.00 hours of readable data
 路 Get transcript length...
 路 Get num feature vectors...
馃槉 Found no audio clips over 30 seconds in length
馃槉 Found no transcripts under 10 characters in length
 路 Get ratio (num_feats / transcript_len)...
馃槉 Found no offending <transcript,clip> pairs
 路 Calculating ratio (num_feats : transcript_len)...
馃槉 Found no <transcript,clip> pairs more than 2.0 standard deviations from the mean
馃帀 鈹? Saved a total of 0.00 hours of data to BEST dataset
   鈹? Removed a total of 0.00 hours (0.00% of original data)
   鈹? Removed a total of 0 samples (0.00% of original data)
   鈹? Wrote best data to /code/data/smoke_test/russian_sample_data/ru.BEST
```

### Run on your data

`data-checker` assumes your CSV has two columns: `wav_filename` and `transcript`. Note that you don't actually need to use WAV files, but the header still should be `wav_filename`.

```
$ docker run data-checker --mount "type=bind,src=/path/to/my/local/data,dst=/mnt" python data_checks.py "/mnt/my-data.csv" 2
```

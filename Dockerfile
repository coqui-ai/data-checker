FROM ghcr.io/coqui-ai/stt-train:main

RUN python -m pip install pandarallel
RUN python -m pip install stt
RUN python -m pip install numpy
RUN python -m pip install librosa

WORKDIR /home/ubuntu
COPY . /home/ubuntu/

RUN python data_checks.py /code/data/smoke_test/ldc93s1_flac.csv 2 /home/ubuntu/stt_model/model.tflite /home/ubuntu/stt_model/huge-vocabulary.scorer
RUN python data_checks.py /code/data/smoke_test/ldc93s1_opus.csv 2 /home/ubuntu/stt_model/model.tflite /home/ubuntu/stt_model/huge-vocabulary.scorer
RUN python data_checks.py /code/data/smoke_test/russian_sample_data/ru.csv 2 /home/ubuntu/stt_model/model.tflite /home/ubuntu/stt_model/huge-vocabulary.scorer

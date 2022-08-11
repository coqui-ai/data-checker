FROM ghcr.io/coqui-ai/stt-train:main

RUN python -m pip install pandarallel
RUN python -m pip install stt
RUN python -m pip install numpy
RUN python -m pip install librosa

WORKDIR /home/ubuntu
COPY . /home/ubuntu/
RUN mkdir /home/ubuntu/stt_model
RUN wget https://github.com/coqui-ai/STT-models/releases/download/english%2Fcoqui%2Fv1.0.0-huge-vocab/model.tflite -P /home/ubuntu/stt_model
RUN wget https://github.com/coqui-ai/STT-models/releases/download/english%2Fcoqui%2Fv1.0.0-huge-vocab/huge-vocabulary.scorer -P /home/ubuntu/stt_model
RUN python data_checks.py /code/data/smoke_test/ldc93s1_flac.csv 2 /home/ubuntu/stt_model/model.tflite /home/ubuntu/stt_model/huge-vocabulary.scorer
RUN python data_checks.py /code/data/smoke_test/ldc93s1_opus.csv 2
RUN python data_checks.py /code/data/smoke_test/russian_sample_data/ru.csv 2 /home/ubuntu/stt_model/model.tflite /home/ubuntu/stt_model/huge-vocabulary.scorer

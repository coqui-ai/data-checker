FROM stt-train-local

RUN python -m pip install pandarallel

WORKDIR /home/ubuntu
COPY . /home/ubuntu/

RUN python data_checks.py /code/data/smoke_test/ldc93s1_flac.csv 2
RUN python data_checks.py /code/data/smoke_test/ldc93s1_opus.csv 2
RUN python data_checks.py /code/data/smoke_test/russian_sample_data/ru.csv 2

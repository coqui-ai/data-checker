FROM local-stt-train

RUN python -m pip install pandarallel

WORKDIR /home/ubuntu
COPY . /home/ubuntu/

RUN python data_checks.py /code/data/smoke_test/russian_sample_data/ru.csv

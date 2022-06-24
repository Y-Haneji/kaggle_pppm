FROM gcr.io/kaggle-gpu-images/python

WORKDIR /home/yuto/PPPM

COPY requirements.txt ./
RUN pip install -U pip &&\
 pip install --no-cache-dir -r requirements.txt

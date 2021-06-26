FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
COPY . /workspace/dcon
RUN apt update
RUN apt install git -y
RUN pip install -e /workspace/dcon/.[deconvolve]

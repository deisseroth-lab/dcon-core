FROM continuumio/miniconda3:4.5.12
RUN mkdir -p /root/.config/matplotlib/stylelib
COPY dark_lab.mplstyle /root/.config/matplotlib/stylelib
COPY . /workspace/dcon
RUN pip install -e /workspace/dcon/.[psfs]

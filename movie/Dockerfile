FROM continuumio/anaconda3:2019.03
RUN mkdir -p /root/.config/matplotlib/stylelib
COPY dark_lab.mplstyle /root/.config/matplotlib/stylelib
COPY . /workspace/dcon
RUN pip install -e /workspace/dcon/.[movie]

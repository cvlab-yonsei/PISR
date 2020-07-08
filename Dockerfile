FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
MAINTAINER Wonkyung Lee <leewk92@yonsei.ac.kr>

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0

RUN pip install matplotlib scikit-image opencv-python tensorboardX easydict

CMD ["/bin/bash"]


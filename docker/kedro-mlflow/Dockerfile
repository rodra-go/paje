FROM python:3.8.5
LABEL MAINTAINER="Rodrigo Cunha <rodrigo.cunha@usp.br>"

RUN apt-get update && \
apt-get upgrade -y

RUN apt-get install bash
RUN apt-get install sudo
RUN apt-get install python3-sphinx -y

ENV GROUP_ID=1000 \
    USER_ID=1000 \
    USER=user01 \
    PASS=pass

ENV PATH="/home/$USER/.local/bin:${PATH}"

RUN addgroup --gid $GROUP_ID $USER
RUN adduser --uid $USER_ID --ingroup $USER --shell /bin/bash --disabled-password --gecos "" $USER
RUN usermod -aG 100 $USER
RUN usermod -aG sudo $USER
RUN echo "$USER:$PASS" | chpasswd
RUN pip install --upgrade pip setuptools

USER $USER
WORKDIR /usr/src/config/

ADD requirements.txt /usr/src/config/
ADD setup.py /usr/src/config/
RUN pip install -r /usr/src/config/requirements.txt --no-warn-script-location
RUN python /usr/src/config/setup.py

WORKDIR /usr/src/code/

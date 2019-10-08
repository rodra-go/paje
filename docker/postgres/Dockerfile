FROM postgres:alpine

# Create sudo user
RUN useradd -ms /bin/bash kong
RUN echo "kong:password" | chpasswd
RUN adduser kong sudo
USER kong
# WORKDIR /home/kong
# Install unzip
# USER root
# RUN apk update && apk add unzip && apk upgrade
COPY get_data.sh /docker-entrypoint-initdb.d/get_data.sh
COPY initdb.sql /docker-entrypoint-initdb.d/initdb.sql
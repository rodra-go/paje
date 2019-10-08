FROM postgres:alpine
# Install unzip
RUN apt-get update && apt-get install -y unzip
COPY get_data.sh /docker-entrypoint-initdb.d/get_data.sh
COPY initdb.sql /docker-entrypoint-initdb.d/initdb.sql
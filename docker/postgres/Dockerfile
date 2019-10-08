FROM postgres:alpine
# Install unzip
RUN apk update && apk add unzip && apk upgrade
COPY get_data.sh /docker-entrypoint-initdb.d/get_data.sh
COPY initdb.sql /docker-entrypoint-initdb.d/initdb.sql
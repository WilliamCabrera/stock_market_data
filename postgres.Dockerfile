# Use the official Ubuntu base image
FROM postgres:16

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip   wget gnupg2 lsb-release libxml2 libssh2-1 nano iputils-ping  && \
    apt-get clean

RUN apt-get update && \
    apt-get install -y sudo

RUN apt install -y postgresql-plpython3-16 nano

RUN usermod -aG sudo postgres

RUN mkdir -p /DB_init_scripts
RUN chown -R postgres:postgres /DB_init_scripts
#COPY create.sql /DB_init_scripts/create.sql

EXPOSE 5432

CMD ["postgres"]
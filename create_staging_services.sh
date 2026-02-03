#!/bin/bash

mkdir data_with_premarket_staging
make build-staging
make start-staging
#docker compose --env-file ./.env.staging  -f docker-compose-staging.yml up
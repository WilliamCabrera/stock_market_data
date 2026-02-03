## steps for first time creating and loading the DB service:

- run the command: `sh create_staging_services.sh`

OR if you want to do it manually:

- make sure the `data_with_premarket_staging` is empty, this will allow to run the scripts in the `docker-entrypoint-initdb.d` folder in the first load

- build the service imagies with: `make build-staging`

- start the services with: `make start-staging`

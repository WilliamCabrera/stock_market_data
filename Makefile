.PHONY:build-staging
build-staging: 
		docker compose --env-file ./.env.staging  -f docker-compose-staging.yml build;

.PHONY: start-staging
start-staging: 
		docker compose --env-file ./.env.staging  -f docker-compose-staging.yml up -d;

.PHONY: stop-staging
stop-staging: 
		docker compose --env-file ./.env.staging  -f docker-compose-staging.yml down;

.PHONY: init-ticker-info
init-ticker-info:
		docker exec -it  python-api-service-staging  python initialization/create_ticker_info_table.py

.PHONY: stock-data-injection
stock-data-injection:
		docker exec -it  python-api-service-staging  python data_injection/injection_stock_data.py

.PHONY: multi-DSMA
multi-DSMA:
		docker exec -it  python-api-service-staging  python data_injection/market_cap_200sma_injection.py

.PHONY: update-process
update-process:
		docker exec -it  python-api-service-staging  python small_caps_strategies/data_update.py




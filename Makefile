# Music DNA Analyzer — lifecycle shortcuts (macOS/Linux/WSL; use run.ps1 on native Windows).
.PHONY: up down logs shell run seed pipeline clean

up:        ## build + start the container in the background
	docker compose up -d --build
down:      ## stop the container
	docker compose down
logs:      ## follow container logs
	docker compose logs -f
shell:     ## open a shell inside the running container
	docker compose exec music-analyzer bash
clean:     ## stop and DELETE the data volume
	docker compose down -v

run:       ## run locally without Docker (venv)
	MUSIC_OPEN_BROWSER=1 python -m app
seed:      ## load demo data
	python seed_demo.py
pipeline:  ## run the full enrich -> parallel-analyze -> lyrics pipeline
	python run_pipeline.py

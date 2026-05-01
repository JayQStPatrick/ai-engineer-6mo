lint:
	ruff check . && black --check .

test:
	pytest -v

run:
	uvicorn app.main:app --reload --port 8000

build:
	docker build -f infra/Dockerfile -t llm-service .

smoke:
	curl -s -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d "{\"text\":\"Alice owes $200 on 2024-03-15\",\"fields\":[\"name\",\"amount\",\"date\"]}"

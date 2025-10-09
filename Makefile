install:
	pip install -r requirements.txt

run:
	docker-compose up -d --build

lint:
	black src

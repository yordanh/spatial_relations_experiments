help:
	@cat Makefile

DATA?="${HOME}/spatial_relations_experiments/"
IMAGE_NAME="spatial_relations_experiments"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
TEST=tests/

build:
	docker build -t $(IMAGE_NAME) -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it --rm -v $(DATA):/data -p 8880:8888 $(IMAGE_NAME) bash
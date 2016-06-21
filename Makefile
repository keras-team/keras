help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) ./nvidia-docker

build:
	docker build -t keras -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data keras bash

ipython: build
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data keras ipython

notebook: build
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data --net=host keras

test: build
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data keras "py.test tests/"


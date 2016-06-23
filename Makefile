help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
TEST=tests/

build:
	docker build -t keras -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras bash

ipython: build
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras ipython

notebook: build
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data --net=host --env KERAS_BACKEND=$(BACKEND) keras

test: build
	$(DOCKER) run -it -v `pwd`:/src -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras py.test $(TEST)


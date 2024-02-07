USER_NAME := perception
TAG_NAME := v0.0.1
IMAGE_NAME := instance_pred
WANDB_API_KEY := $(shell echo $$WANDB_API_KEY)

UID := 1000
GID := 1000


define run_docker
	clear && \
	docker run -it --rm \
		--net host \
		--gpus all \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name=$(IMAGE_NAME)_container \
		-u $(USER_NAME) \
		-v ./:/home/$(USER_NAME)/workspace \
		-v /home/robesafe/nuscenes/:/home/$(USER_NAME)/Datasets/nuscenes \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		'$(IMAGE_NAME)':$(TAG_NAME) \
		/bin/bash -c $(1)
endef

define run_docker_dgx
	docker run -it --rm \
		--net host \
		--gpus '"device=4,5"'\
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name=$(IMAGE_NAME)_container \
		-u $(USER_NAME) \
		-v /home/miguel.antunes/workspace:/home/$(USER_NAME)/workspace \
		-v /home/miguel.antunes/nuscenes/:/home/$(USER_NAME)/Datasets/nuscenes \
		$(IMAGE_NAME):$(TAG_NAME) \
		/bin/bash -c $(1)
endef

build:
	docker build ./docker -t '$(IMAGE_NAME)':$(TAG_NAME) --force-rm --build-arg USER=$(USER_NAME) --build-arg USER_ID=$(UID) --build-arg USER_GID=$(GID)

attach:
	docker exec -it $(IMAGE_NAME)_container bash

run:
	$(call run_docker, "bash")

run_dgx:
	$(call run_docker_dgx, "bash")


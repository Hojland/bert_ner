commit_hash=$(shell git rev-parse HEAD)
project=$(notdir $(shell pwd))
image_name=docker.pkg.github.com/nuuday/bert_email_router/bert_email_router
USERNAME=$(shell awk -F "=" '/USERNAME/{print $2}' .jenkins.env)
USER_TOKEN=$(shell awk -F "=" '/USER_TOKEN/{print $2}' .jenkins.env)
BUILD_TOKEN=$(shell awk -F "=" '/BUILD_TOKEN/{print $2}' .jenkins.env)


.PHONY: build_local_cpu, build_local_gpu, run_local, run_jupyter

build_local_cpu:
	docker build -t ${image_name}:local-cpu --build-arg COMMIT_HASH=${commit_hash} --build-arg PROD_ENV=$(env) \
		--build-arg COMPUTE_KERNEL=cpu --build-arg IMAGE_NAME=python:3.8 -f Dockerfile .

build_local_gpu:
	make docker_login
	docker build -t ${image_name}:local-gpu --build-arg COMMIT_HASH=${commit_hash} --build-arg PROD_ENV=$(env) \
		--build-arg COMPUTE_KERNEL=gpu -f Dockerfile .
	docker push ${image_name}:local-gpu
	

run_train:
	@docker run \
		-it \
		-d \
		--rm \
		--gpus all \
		--name $(project)-train \
		--env-file .env \
		${image_name}:local-gpu \
		"python3 train.py"

run_app:
	@docker run \
		-it \
		-d \
		--rm \
		-p 8000:8000 \
		--name $(project)-api \
		--env-file .env \
		${image_name}:local-cpu \
		"uvicorn app:app --host 0.0.0.0 --port 8000"

dev_gpu:	
	@$(MAKE) -s stop

	@$(MAKE) -s docker_login

	@docker run \
		-it \
		-d \
		-p 8888:8888 \
		-p 8787:8787 \
		--rm \
		--gpus all \
		--name $(project) \
		--env-file .env \
		-v $(PWD):/app/ \
		$(image_name):local-gpu bash > /dev/null
	
	@docker exec -it -d $(project) bash \
		-c "jupyter lab --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=$(token)"

	@echo "Container started"
	@echo "Jupyter Lab is running at http://localhost:8888/?token=ml"

dev_cpu:	
	@$(MAKE) -s stop

	@$(MAKE) -s docker_login

	@docker run \
		-it \
		-d \
		-p 8888:8888 \
		-p 8787:8787 \
		--rm \
		--name $(project) \
		--env-file .env \
		-v $(PWD):/app/ \
		$(image_name):local-cpu bash > /dev/null
	
	@docker exec -it -d $(project) bash \
		-c "jupyter lab --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=$(token)"

	@echo "Container started"
	@echo "Jupyter Lab is running at http://localhost:8888/?token=ml"
stop:
	@docker stop $(project) > /dev/null 2>&1 ||:
	@docker container prune --force > /dev/null

docker_login:
	@echo "Requesting credentials for docker login"
	@$(eval export GITHUB_ACTOR=nuuday)
	@$(eval export GITHUB_TOKEN=$(shell awk -F "=" '/GITHUB_TOKEN/{print $$NF}' .env))
	@docker login https://docker.pkg.github.com/nuuday/ -u $(GITHUB_ACTOR) -p $(GITHUB_TOKEN)

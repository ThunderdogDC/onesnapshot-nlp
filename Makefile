.PHONY: clean environment data install_git_hooks linux_requirements python_requirements requirements precommit test conda_cache_s3 conda_check_cache cruft_cleanup

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = onesnapshot-nlp
SENSITIVE_PROJECT = yes
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python
PIP_VERSION = 25.3

NOW:=$(shell date +"%m-%d-%y_%H-%M-%S")


#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Set up python interpreter environment
environment:
	conda env remove --name $(PROJECT_NAME) || true
ifneq ("$(wildcard repo_settings/conda.lock.yml)","") 
	@echo ">>> Creating conda environment from conda lock file"
	conda env create -f repo_settings/conda.lock.yml
else
	@echo ">>> Creating conda environment from scratch"
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) pip=$(PIP_VERSION) setuptools wheel pip-tools
endif
	@echo ">>> New conda env created. Activate with: \n conda activate $(PROJECT_NAME)"

## Generate extract of dataset
data:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py --output_name extract__$(NOW)


## Reformat, lint
precommit:
	@git diff --cached --name-only --diff-filter=ACM | grep -E '\.(py|ipynb)$$' | xargs -r ruff check --fix 
	@git diff --cached --name-only --diff-filter=ACM | grep -E '\.(py|ipynb)$$' | xargs -r ruff format

## Install Linux dependencies
linux_requirements:
	@echo "No Linux requirements to install"


## Install Python Dependencies
python_requirements:
	$(PYTHON_INTERPRETER) -m piptools compile --extra dev --resolver=backtracking --output-file=repo_settings/auto_reqs.txt pyproject.toml
	$(PYTHON_INTERPRETER) -m piptools sync repo_settings/auto_reqs.txt
	$(PYTHON_INTERPRETER) -m pip install --editable .
	$(PYTHON_INTERPRETER) -m ipykernel install --user --name=$(PROJECT_NAME)
	pre-commit install --config repo_settings/.pre-commit-config.yaml
	# For moving between tools, clear cache
	pre-commit clean


## Install combined dependencies
requirements:
	make linux_requirements
	make python_requirements


## Run Python tests
test:
	pytest test


## Cache conda env to S3
conda_cache_s3:
	@echo Uploading cached env to S3
	$(PYTHON_INTERPRETER) repo_settings/.cache.py upload $(PROJECT_NAME) CondaEnv/$(PROJECT_NAME).tar

## Check S3 for cached conda env
conda_check_cache:
	@echo Checking if $(PROJECT_NAME) has cached env on S3
	$(PYTHON_INTERPRETER) repo_settings/.cache.py check $(PROJECT_NAME).tar

## Download cached conda env from S3
get_conda_cache:
	@echo Getting cached env from S3
	$(PYTHON_INTERPRETER) repo_settings/.cache.py download $(PROJECT_NAME) CondaEnv/$(PROJECT_NAME).tar

# Tidy directories post cruft update
cruft_cleanup:
	# Checking if any rejected files post patch
	@if [ -n "$$(find . -name "*.rej")" ]; then \
		echo "Aborting. Please resolve conflicts before running cleanup. The following .rej files were found:"; \
		find . -name "*.rej"; \
		exit 1; \
	else \
		if [ "no" = "yes" ]; then \
			$(PYTHON_INTERPRETER) repo_settings/.setup_visualisation.py; \
			rm -f repo_settings/.setup_visualisation.py; \
		else \
			rm -f app; \
			rm -f Dockerfile; \
		fi; \
		if [ "yes" = "yes" ]; then \
			echo "*.ipynb filter=strip-nb-output" > .gitattributes; \
		else \
			echo "project configured as non-sensitive: notebook outputs will not be cleared on commit"; \
		fi; \
		if [ ! "no" = "yes" ]; then \
			rm -f .gitlab-ci.yml; \
			rm -f Dockerfile.ci; \
		fi; fi; git add .; \
		git commit -m "update cookiecutter template"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

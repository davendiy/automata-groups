
pure-install:
	python -m venv venv
	./venv/bin/python -m pip install Cython
	./venv/bin/python -m pip install -r requirements.txt
	./venv/bin/python -m pip install .

lint:
	isort autogrp
	isort tests
	flake8 autogrp

clean:
	mkdir build || true
	mkdir build/_autogrp_cython/ || true
	for file in _autogrp_cython/*.c ; do \
		echo "moving $${file} ..." ; \
		mv "$$file" "build/$${file}" ; \
	done \

install: pure-install clean


# In addition, as explained by skwllsp, you need to tell make to execute the command
# list for each target as a single shell script rather than line by line,
# which you can do in GNU make by defining a .ONESHELL target.
.ONESHELL:

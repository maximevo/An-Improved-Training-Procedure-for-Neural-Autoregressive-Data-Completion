all: linalg_.so nonlinear_.so 

linalg_.so: linalg_.h linalg_.c
	python setup.py install --install-platlib=.

nonlinear_.so: nonlinear_.h nonlinear_.c
	python setup.py install --install-platlib=.

clean:
	rm -f linalg_.so nonlinear_.so MLPythonMath*
	rm -rf build


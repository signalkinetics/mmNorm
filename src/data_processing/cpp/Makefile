LDLIBS=-lboost_system -lboost_thread 
LDINCLUDES=`python3 -m pybind11 --includes` \
			-I../ \
			-I/usr/local/include \
			-I/usr/local/lib \
			-I../../utils/include \
			
CPPFLAGS=-Wall -Ofast -std=c++14 -fPIC -shared -march=native -frename-registers -fopenmp
PYTHON_EXT=`python3-config --extension-suffix`

all: imaging

imaging: imaging.cpp
	g++ $(CPPFLAGS) imaging.cpp -o imaging$(PYTHON_EXT) $(LDINCLUDES) $(LDLIBS)
CC = g++ -g
NVCC = nvcc -ccbin gcc
CFLAGS = -O2 -c -Wall -pedantic
CFLAGS_GPU = -O2 -c -g --ptxas-options=-v
LDFLAGS = -lpthread -lboost_thread -lboost_system

GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)

openglGPU: openglcppGPU ScalarFunctionGPU gpu
	nvcc opengl.o ScalarFunction.o gpu.o -o openglGPU -lSDL -lGLU -lGL -lGLEW $(LDFLAGS)

opengl: openglcpp ScalarFunction
	$(CC) opengl.o ScalarFunction.o -o opengl -lSDL -lGLU -lGL -lGLEW $(LDFLAGS)

openglcpp:
	$(CC) $(CFLAGS) opengl.cpp

openglcppGPU:
	$(CC) $(CFLAGS) -DGPU opengl.cpp

test: test_cpp Image convexHull ScalarFunction
	$(CC) test.o Image.o convexHull.o ScalarFunction.o -o test -lpng $(LDFLAGS)

test_cpp : test.cpp
	$(CC) $(CFLAGS) test.cpp 

Image: Image.cpp
	$(CC) $(CFLAGS) Image.cpp

convexHull: convexHull.cpp
	$(CC) $(CFLAGS) convexHull.cpp

ScalarFunction: ScalarFunction.cpp
	$(CC) $(CFLAGS) ScalarFunction.cpp

gpu: 
	$(NVCC) $(CFLAGS_GPU) $(GENCODE_FLAGS) gpu.cu

ScalarFunctionGPU: ScalarFunction.cpp
	$(CC) $(CFLAGS) -DGPU ScalarFunction.cpp

clean:
	rm *.o
	rm test
	rm opengl

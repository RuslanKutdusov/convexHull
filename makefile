CC = g++ -g
NVCC = nvcc -ccbin gcc
CFLAGS = -O2 -c -Wall -pedantic
CFLAGS_GPU = -O2 -c -g --ptxas-options=-v
LDFLAGS = -lpthread -lboost_thread -lboost_system

GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)

openglGPU: ScalarFunctionGPU gpu
	$(CC) $(CFLAGS) -DGPU opengl.cpp
	nvcc opengl.o ScalarFunction.o gpu.o -o openglGPU -lSDL -lGLU -lGL -lGLEW $(LDFLAGS)

opengl: ScalarFunction
	$(CC) $(CFLAGS) opengl.cpp
	$(CC) opengl.o ScalarFunction.o -o opengl -lSDL -lGLU -lGL -lGLEW $(LDFLAGS)

test: ScalarFunctionGPU gpu
	$(CC) $(CFLAGS) -DGPU test.cpp 
	nvcc test.o ScalarFunction.o gpu.o -o test -lpng $(LDFLAGS)

ScalarFunction: ScalarFunction.cpp
	$(CC) $(CFLAGS) ScalarFunction.cpp

ScalarFunctionGPU: ScalarFunction.cpp
	$(CC) $(CFLAGS) -DGPU ScalarFunction.cpp

gpu: 
	$(NVCC) $(CFLAGS_GPU) $(GENCODE_FLAGS) gpu.cu

clean:
	rm *.o
	rm test
	rm opengl
	rm openglGPU

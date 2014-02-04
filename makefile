CC = g++ -g
NVCC = nvcc
CFLAGS = $(BOOST_HEADERS) -O2 -c -Wall -pedantic
CFLAGS_GPU = -ccbin gcc $(BOOST_HEADERS) -O2 -c -g --ptxas-options=-v
LDFLAGS = $(BOOST_LD_PATH) -lpthread -lboost_thread -lboost_system -lboost_serialization

GENCODE_SM10    := -gencode arch=compute_12,code=sm_12
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20)

vis: ScalarFunction
	$(CC) $(CFLAGS) vis.cpp
	$(CC) vis.o ScalarFunction.o -o vis -lSDL -lGLU -lGL -lGLEW $(LDFLAGS)

test: ScalarFunctionGPU gpu
	$(CC) $(CFLAGS) -DGPU test.cpp 
	$(NVCC) test.o ScalarFunction.o gpu.o -o test -lpng $(LDFLAGS)

time_measurements: ScalarFunctionGPU gpu
	$(CC) $(CFLAGS) -DGPU time_measurements.cpp 
	$(NVCC) time_measurements.o ScalarFunction.o gpu.o -o time_measurements $(LDFLAGS)

ScalarFunction: ScalarFunction.cpp
	$(CC) $(CFLAGS) ScalarFunction.cpp

ScalarFunctionGPU: ScalarFunction.cpp
	$(CC) $(CFLAGS) -DGPU ScalarFunction.cpp

gpu: 
	$(NVCC) $(CFLAGS_GPU) $(GENCODE_FLAGS) gpu.cu

clean:
	rm *.o
	rm test
	rm vis
	rm time_measurements

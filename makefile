CC = g++
NVCC = /opt/cuda-5.5/bin/nvcc
PRECISION = -DDOUBLE_PRECISION
CFLAGS = $(BOOST_HEADERS) -O2 -c -Wall -pedantic $(PRECISION)
CFLAGS_GPU = -ccbin gcc $(BOOST_HEADERS) -O2 -c --ptxas-options=-v $(PRECISION)
LDFLAGS = $(BOOST_LD_PATH) -lpthread -lboost_thread -lboost_system -lboost_serialization

GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_FLAGS   := $(GENCODE_SM20)

vis: ScalarFunction.o vis.o
	$(CC) vis.o ScalarFunction.o -o vis -lSDL -lGLU -lGL -lGLEW $(LDFLAGS)

vis.o: vis.cpp
	$(CC) $(CFLAGS) vis.cpp -o vis.o

test: ScalarFunction.o ScalarFunctionGPU.o test.o
	$(NVCC) test.o ScalarFunctionGPU.o ScalarFunction.o -o test $(LDFLAGS)

test.o: test.cpp
	$(CC) $(CFLAGS) -DGPU test.cpp -o test.o

time_measurements: ScalarFunction.o ScalarFunctionGPU.o time_measurements.o
	$(NVCC) time_measurements.o ScalarFunction.o ScalarFunctionGPU.o -o time_measurements $(LDFLAGS)

time_measurements.o: time_measurements.cpp
	$(CC) $(CFLAGS) -DGPU time_measurements.cpp -o time_measurements.o

ScalarFunction.o: ScalarFunction.cpp
	$(CC) $(CFLAGS) -DGPU ScalarFunction.cpp -o ScalarFunction.o

ScalarFunctionGPU.o: ScalarFunction.cu
	$(NVCC) $(CFLAGS_GPU) $(GENCODE_FLAGS) -DGPU ScalarFunction.cu -o ScalarFunctionGPU.o

access: access.o
	$(NVCC) access.o -o access

access.o: access.cu
	$(NVCC) $(CFLAGS_GPU) $(GENCODE_FLAGS) access.cu -o access.o

bandwidth: bandwidth.o
	$(NVCC) bandwidth.o -o bandwidth	

bandwidth.o: bandwidth.cu
	$(NVCC) $(CFLAGS_GPU) $(GENCODE_FLAGS) bandwidth.cu -o bandwidth.o

clean:
	rm *.o
	rm test
	rm vis
	rm time_measurements
	rm access
	rm bandwidth

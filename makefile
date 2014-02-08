CC = g++ -g
NVCC = nvcc
CFLAGS = $(BOOST_HEADERS) -O2 -c -Wall -pedantic
CFLAGS_GPU = -ccbin gcc $(BOOST_HEADERS) -O2 -c -g --ptxas-options=-v
LDFLAGS = $(BOOST_LD_PATH) -lpthread -lboost_thread -lboost_system -lboost_serialization

GENCODE_SM10    := -gencode arch=compute_12,code=sm_12
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20)

vis: ScalarFunction.o vis.o
	$(CC) vis.o ScalarFunction.o -o vis -lSDL -lGLU -lGL -lGLEW $(LDFLAGS)

vis.o: vis.cpp
	$(CC) $(CFLAGS) vis.cpp -o vis.o

test: ScalarFunctionGPU.o gpu.o test.o
	$(NVCC) test.o ScalarFunctionGPU.o gpu.o -o test -lpng $(LDFLAGS)

test.o: test.cpp
	$(CC) $(CFLAGS) -DGPU test.cpp -o test.o

time_measurements: ScalarFunctionGPU.o gpu.o time_measurements.o
	$(NVCC) time_measurements.o ScalarFunctionGPU.o gpu.o -o time_measurements $(LDFLAGS)

time_measurements.o: time_measurements.cpp
	$(CC) $(CFLAGS) -DGPU time_measurements.cpp -o time_measurements.o

ScalarFunction.o: ScalarFunction.cpp
	$(CC) $(CFLAGS) ScalarFunction.cpp -o ScalarFunction.o

ScalarFunctionGPU.o: ScalarFunction.cpp
	$(CC) $(CFLAGS) -DGPU ScalarFunction.cpp -o ScalarFunctionGPU.o

gpu.o: gpu.cu
	$(NVCC) $(CFLAGS_GPU) $(GENCODE_FLAGS) gpu.cu -o gpu.o

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

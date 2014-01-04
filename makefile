CFLAGS = -O2 -c -Wall -pedantic

test: test_cpp Image convexHull ScalarFunction
	clang++ -lpng test.o Image.o convexHull.o ScalarFunction.o -o test

test_cpp : test.cpp
	clang++ $(CFLAGS) test.cpp 

Image: Image.cpp
	clang++ $(CFLAGS) Image.cpp

convexHull: convexHull.cpp
	clang++ $(CFLAGS) convexHull.cpp

ScalarFunction: ScalarFunction.cpp
	clang++ $(CFLAGS) ScalarFunction.cpp

clean:
	rm *.o
	rm test

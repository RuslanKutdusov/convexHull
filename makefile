CFLAGS = -O2 -c -Wall -pedantic
LDFLAGS = -lpthread -lboost_thread -lboost_system

opengl: openglcpp ScalarFunction
	clang++ opengl.o ScalarFunction.o -o opengl -lSDL -lGLU -lGL -lGLEW $(LDFLAGS)

openglcpp:
	clang++ $(CFLAGS) opengl.cpp

test: test_cpp Image convexHull ScalarFunction
	clang++ test.o Image.o convexHull.o ScalarFunction.o -o test -lpng $(LDFLAGS)

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
	rm opengl

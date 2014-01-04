CFLAGS = -O2 -c -Wall -pedantic

opengl: openglcpp ScalarFunction
	clang++ opengl.o ScalarFunction.o -o opengl -lSDL -lGLU -lGL -lGLEW

openglcpp:
	clang++ $(CFLAGS) opengl.cpp

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
	rm opengl

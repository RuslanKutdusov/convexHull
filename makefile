CFLAGS = -O2 -c -Wall -pedantic

test: test_cpp Image
	clang++ -lpng test.o Image.o -o test

test_cpp : test.cpp
	clang++ $(CFLAGS) test.cpp 

Image: Image.cpp
	clang++ $(CFLAGS) Image.cpp

clean:
	rm *.o
	rm test
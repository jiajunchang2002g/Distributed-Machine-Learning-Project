CPP=mpicxx
CPPFLAGS= -O3

.PHONY: all clean

all: engine engine.debug

clean:
	rm -rf engine engine.debug

engine: engine.cpp common.cpp
	${CPP} ${CPPFLAGS} $^ -o $@

engine.debug: engine.cpp common.cpp
	${CPP} ${CPPFLAGS} -g -Wall -DDEBUG $^ -o $@

engine.debug1: engine.cpp common.cpp
	${CPP} ${CPPFLAGS} -g -Wall -DDEBUG -DDEBUG1 $^ -o $@

# debug target
debug: engine.debug
	@echo "Debug build complete: engine.debug"

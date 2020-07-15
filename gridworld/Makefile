CC = g++
# --verbose

OPT_FLAGS = -o3 -std=gnu++11 -Wno-delete-non-virtual-dtor 
DEBUG_FLAGS = -g -std=gnu++11 -Wno-delete-non-virtual-dtor

# CFLAGS = ${DEBUG_FLAGS}
CFLAGS = ${OPT_FLAGS}

HEADERS = headers/
ZAFFRE = zaffre/
EIGEN = eigen/

SRCDIR:=src
OBJDIR:=obj

SRC := $(shell find $(SRCDIR) -name "[!p]*.cpp")
OBJ:= $(shell find $(OBJDIR) -name "*.o")

INCLUDES = -I${HEADERS} -I${ZAFFRE} -I${EIGEN} 

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))

ZAFFRE_SRC = ${ZAFFRE}/Zaffre/src/IOStringUtils.cpp ${ZAFFRE}/Zaffre/src/MathUtils.cpp ${ZAFFRE}/Zaffre/src/MarsagliaGenerator.cpp ${ZAFFRE}/Zaffre/src/tinv.cpp ${ZAFFRE}/Zaffre/src/LinearAlgebra.cpp


#.PHONY: all clean

all: obj/Zaffre.o obj/tinv.o obj/MathUtils.o obj/MarsagliaGenerator.o obj/IOStringUtils.o ${OBJ_FILES} main

main: ${OBJ}
	${CC} ${INCLUDES} ${CFLAGS} -o main ${OBJ} -lprotobuf;

obj/%.o: src/%.cpp
	${CC} ${INCLUDES} $(CFLAGS) -c -o $@ $<

obj/Zaffre.o: ${ZAFFRE}/Zaffre/src/Zaffre.cpp obj/IOStringUtils.o
	${CC} ${INCLUDES} ${CFLAGS} -c ${ZAFFRE}/Zaffre/src/Zaffre.cpp -o obj/Zaffre.o 

obj/MarsagliaGenerator.o: ${ZAFFRE}/Zaffre/src/MarsagliaGenerator.cpp
	${CC} ${INCLUDES} ${CFLAGS} -c ${ZAFFRE}/Zaffre/src/MarsagliaGenerator.cpp -o obj/MarsagliaGenerator.o

obj/tinv.o: ${ZAFFRE}/Zaffre/src/tinv.cpp obj/IOStringUtils.o obj/LinearAlgebra.o
	${CC} ${INCLUDES} ${CFLAGS} -c ${ZAFFRE}/Zaffre/src/tinv.cpp -o obj/tinv.o

obj/MathUtils.o: ${ZAFFRE}/Zaffre/src/MathUtils.cpp
	${CC} ${INCLUDES} ${CFLAGS} -c ${ZAFFRE}/Zaffre/src/MathUtils.cpp -o obj/MathUtils.o

obj/IOStringUtils.o: ${ZAFFRE}/Zaffre/src/IOStringUtils.cpp
	${CC} ${INCLUDES} ${CFLAGS} -c ${ZAFFRE}/Zaffre/src/IOStringUtils.cpp -o obj/IOStringUtils.o

obj/LinearAlgebra.o: ${ZAFFRE}/Zaffre/src/LinearAlgebra.cpp
	${CC} ${INCLUDES} ${CFLAGS} -c ${ZAFFRE}/Zaffre/src/LinearAlgebra.cpp -o obj/LinearAlgebra.o

clean:
	rm -f main obj/*

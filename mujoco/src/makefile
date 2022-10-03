
#LINUX
COMMON=-O2 -I"./mujoco-2.2.2/include" -L"./mujoco-2.2.2/lib" -pthread -Wl,-no-as-needed -Wl,-rpath,'$$ORIGIN'/mujoco-2.2.2//lib
LIBS = -lmujoco -lglfw
CC = gcc


ROOT = main

all:
	$(CC) $(COMMON) main.c $(LIBS) -o $(ROOT)

main.o:
	$(CC) $(COMMON) -c main.c

clean:
	rm *.o $(ROOT)

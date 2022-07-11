#targets

FLAGS = $(pkg-config --cflags --libs opencv4)
COMPILE = nvcc
FILE_NAME = main.cu
FILE_NAME_OUT= gray

All:
	$(COMPILE) $(FILE_NAME) -o $(FILE_NAME_OUT) $(FLAGS)
Clean:
	rm-f $(FILE_NAME_OUT)


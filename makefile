# compiler
#CC = g++
NVCC = nvcc
CC = g++
CC_FLAGS = -c
CC_LIBS = cuda cudart cufft opencv_core opencv_highgui opencv_imgproc opencv_videoio
CC_LIBS_PARAMS = $(foreach d,$(CC_LIBS),-l$d)

# other project files, .o files,  header files
VPATH = /src
SRC = cuda.cu video_param.h
OBJ_DIR = obj
INC_DIR = inc
LIB = /usr/local/cuda-10.1/targets/x86_64-linux/lib/
INC = inc /usr/local/cuda-10.1/lib64 /usr/local/include/opencv4 /usr/local/cuda-10.1/targets/x86_64-linux/lib/
INC_PARAMS = $(foreach d,$(INC),-I$d)

# print structuring
PRINT = @echo "\nBuilding: "$@

EXE = aahp

# build
# $(EXE): cuda.o main.o
# 	$(PRINT)
# 	$(CC) -L$(LIB) $(INC_PARAMS) $(CC_LIBS_PARAMS) obj/cuda.o -o $@

# build
$(EXE): main.o cuda.o
	$(CC) $(CC_FLAGS) $(INC_PARAMS) obj/main.o obj/cuda.o -o  $@

main.o:
	$(PRINT)
	$(NVCC) $(CC_FLAGS) $(INC_PARAMS) -L$(LIB) $(CC_LIBS_PARAMS) -c src/main.cu -o $(OBJ_DIR)/$@

cuda.o:
	$(PRINT)
	$(NVCC) $(CC_FLAGS) $(INC_PARAMS) -L$(LIB) $(CC_LIBS_PARAMS) -c src/cuda.cu -o $(OBJ_DIR)/$@

clean:
	rm obj/*.o

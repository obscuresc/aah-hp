# compiler
#CC = g++
NVCC = nvcc
CC = g++
CC_FLAGS = -c
CC_LIBS = -lcuda -lcudart -lcufft -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio

# other project files, .o files,  header files
VPATH = /src
SRC = cuda.cu video_param.h
OBJ_DIR = obj
INC_DIR = include
CUDA = /usr/local/cuda-10.1/include
CV = /usr/local/include/opencv4

# object files:
OBJS = cuda.o

EXE = aahp

# build
$(EXE): cuda video_param
	$(NVCC) -o $@ -I$(CV) -I$(CUDA) $(CC_LIBS) obj/$(OBJS) obj/video_param.o

# main
main: $(OBJS)
	$(NVCC) $(CC_FLAGS) $(CC_LIBS) $@

cuda:
	$(NVCC) $(CC_FLAGS) -I$(CUDA) -I$(CV) $(CC_LIBS) -c src/$@.cu -o $(OBJ_DIR)/$@.o

video_param:
	$(CC) $(CC_FLAGS) -I$(CUDA) -I$(CV) $(CC_LIBS) -c src/$@.h -o $(OBJ_DIR)/$@.o


clean:
	rm obj/* *.o

# main
#main:
#	$(CC) -c $(OBJS) $@

# cuda
#cuda:
#	$(CC) $(CC_FLAGS) $(CC_LIBS) $(SRC)/cuda.cu -I$(CUDA) -I$(CV) $@

# build
#$(EXE) : $(OBJS)
#	$(CC) $(CC_FLAGS) -lcuda -lcudart -lcufft -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio $(OBJS) -o  $@
#
# compile main .cpp file to object files:
#$(OBJ_DIR)/%.o : %.cpp
#	$(CC) $(CC_FLAGS) -lcuda -lcudart -lcufft -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -I$(CU_INC_DIR) -I$(CV_INC_DIR) -c $< -o $@

# compile C++ source files to object files:
#$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(CU_INC_DIR)/%.h $(CV_INC_DIR)/%.h
#	$(CC) $(CC_FLAGS) -c $< -o $@

# compile cuda source files to object files:
#$(OBJ_DIR)/%.cu : $(SRC_DIR)/%.cu $(CU_INC_DIR)/%.h $(CV_INC_DIR)/%.h
#	$(CC) $(CC_FLAGS) -lcuda -lcudart -lcufft -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -c $< -o $@

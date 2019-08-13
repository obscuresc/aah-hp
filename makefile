# compiler
#CC = g++
CC = nvcc
CC_FLAGS =
CC_LIBS =

# other project files, .o files,  header files
SRC_DIR = src
OBJ_DIR = obj
INC_DIR = include
CU_INC_DIR = /usr/local/cuda-10.1/include

EXE = aah-hp

# Object files:
OBJS = $(OBJ_DIR)/main.o

# build
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) -lcuda -lcufft $(OBJS) -o  $@

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -I$(CU_INC_DIR) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(CU_INC_DIR)/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

clean:
	$(RM) obj/* *.o

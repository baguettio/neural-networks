# Specify the compiler
CXX = g++

# Specify options for the compiler
CXXFLAGS = -Wall -Wextra -std=c++11 -O3

# Target executable name
TARGET = a.out

# .cpp files
SRCS = main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default rule (first one in the Makefile)
all: $(TARGET)

# Rule to link object files to produce target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for cleaning up object files and executable
clean:
	rm -f $(OBJS) $(TARGET)

# Declare the "clean" target as phony (not a file)
.PHONY: clean all

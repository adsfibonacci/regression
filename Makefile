CXX = g++
CXXFLAGS = -I$(GUROBI_HOME)/include -Wall -Werror
# LDFLAGS = -L$(GUROBI_HOME)/lib -lgurobi_c++ -lgurobi120

TARGET = test
SRC = test.cpp kfolds.cpp load_files.cpp logistic_regression.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJ)

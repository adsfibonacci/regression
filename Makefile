CXX = g++
CXXFLAGS = -Wall -Werror
# LDFLAGS = -L$(GUROBI_HOME)/lib -lgurobi_c++ -lgurobi120

TARGET = test
SRC = test.cpp kfolds.cpp load_files.cpp logistic_regression.cpp scoring.cpp regressor.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET) 

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

check-syntax:
	$(CC) -Wall -Wextra -pedantic -fsyntax-only $(CHK_SOURCES) || exit 0

clean:
	rm -f $(TARGET) $(OBJ)

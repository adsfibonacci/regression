CXX = g++
CXXFLAGS = -I$(GUROBI_HOME)/include -Wall -Werror
# LDFLAGS = -L$(GUROBI_HOME)/lib -lgurobi_c++ -lgurobi120

TARGET = test
SRC = test.cpp kfolds.cpp load_files.cpp logistic_regression.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

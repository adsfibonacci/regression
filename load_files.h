#ifndef LOADER_H
#define LOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Read a 2D matrix from file
std::vector<std::vector<double>> loadMatrix(const std::string &filename);
std::vector<double> loadVector(const std::string& filename);
#endif

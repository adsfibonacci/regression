#include "load_files.h"

// Read a 2D matrix from file
std::vector<std::vector<double>> loadMatrix(const std::string& filename) {
  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Could not open file " + filename);
  
  // Read dimensions
  double rows, cols;
  file >> rows >> cols;
  
  // Skip to the '-' line
  std::string line;
  while (std::getline(file, line)) {
    if (line.find('-') != std::string::npos) break;
  }
  
  // Read data into matrix
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      file >> matrix[i][j];
    }
  }
  return matrix;
}

// Read a column vector from file
std::vector<double> loadVector(const std::string& filename) {
  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Could not open file " + filename);
  
  // Read dimension
  double rows;
  file >> rows;
  
  // Skip to the '-' line
  std::string line;
  while (std::getline(file, line)) {
    if (line.find('-') != std::string::npos) break;
  }
  
  // Read data into vector
  std::vector<double> vec(rows);
  for (size_t i = 0; i < rows; ++i) {
    file >> vec[i];
  }
  return vec;
}


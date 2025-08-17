#include "load_files.h"

std::vector<std::vector<double>> loadMatrix(const std::string& filename) {
  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Could not open file " + filename);
  double rows, cols;
  file >> rows >> cols;

  // "-" is the seperation between the header and the data
  std::string line;
  while (std::getline(file, line)) {
    if (line.find('-') != std::string::npos) break;
  }
  
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      file >> matrix[i][j];
    }
  }
  return matrix;
}

std::vector<double> loadVector(const std::string& filename) {
  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Could not open file " + filename);

  double rows;
  file >> rows;

  // "-" is the seperation between the header and the data
  std::string line;
  while (std::getline(file, line)) {
    if (line.find('-') != std::string::npos) break;
  }
  
  std::vector<double> vec(rows);
  for (size_t i = 0; i < rows; ++i) {
    file >> vec[i];
  }
  return vec;
}


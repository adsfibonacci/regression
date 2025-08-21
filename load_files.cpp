#include "load_files.h"

std::vector<std::vector<double>> loadMatrix(const std::string& filename, bool norm) {
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
  if (norm)
    normalize(matrix);
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

void normalize(std::vector<std::vector<double>>& X) {
  size_t n = X.size(), d = X[0].size();
  for (size_t j = 0; j < d; ++j) {
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i)
      mean += X[i][j];
    mean /= n;
    
    double var = 0.0;
    for (size_t i = 0; i < n; ++i)
      var += (X[i][j] - mean) * (X[i][j] - mean);
    double stddev = std::sqrt(var / n);
    if (stddev == 0) stddev = 1.0; // prevent div by zero
    
    for (size_t i = 0; i < n; ++i)
      X[i][j] = (X[i][j] - mean) / stddev;
  }
}


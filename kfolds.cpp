#include "kfolds.h"
using std::cout, std::endl;

template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> &v) {
  for (size_t i = 0; i < v.size(); ++i)
    os << v[i] << "\t";
  os << endl;  
  return os;
}  

/**
 * Return N-folds of indices to define the X_train/test and y_train/test
 * samples should be the first dimension of the design matrix
 * k should be the number of folds to make
 * seed is the random state
 */
std::vector<std::vector<size_t>> kfolds(size_t samples, size_t k,
                                        unsigned int seed) {
  std::vector<size_t> indices(samples);
  std::iota(indices.begin(), indices.end(), 0);

  std::mt19937 rng(seed);
  std::shuffle(indices.begin(), indices.end(), rng);
  
  std::vector<std::vector<size_t>> kfolds(k);
  size_t start = 0;
  for (size_t i = 0; i < k; ++i) {
    size_t size = (samples / k) + ((i < samples % k) ? 1 : 0);    
    std::vector<size_t> fold(indices.begin() + start,
                             indices.begin() + start + size);
    kfolds[i] = fold;
    start += size;
  }    
  return kfolds;
}

/**
 * Return N-folds of indices to define the X_train/test and y_train/test in a stratified manner
 * y should be the response vector with all response classes
 * k should be the number of folds to make
 * seed is the random state
 */
std::vector<std::vector<size_t>>
stratified_kfolds(std::vector<double> &y, size_t k, unsigned int seed) {
  std::unordered_map<double, std::vector<size_t>> class_indices;
  for (size_t i = 0; i < y.size(); ++i)
    class_indices[y[i]].push_back(i);
  
  std::mt19937 rng(seed);
  std::vector<size_t> combined_indices;
  
  bool remaining = true;
  std::unordered_map<double, size_t> cursors;
  for (auto& [c, indices] : class_indices) {
    std::shuffle(indices.begin(), indices.end(), rng);
    cursors[c] = 0;
  }
  while (true) {
    remaining = false;
    for (auto& [c, indices] : class_indices) {
      if (cursors[c] < indices.size()) {
	combined_indices.push_back(indices[cursors[c]]);
	cursors[c]++;
	remaining = true;
      }
    }
    if (!remaining)
      break;
  }
  
  std::vector<std::vector<size_t>> kfolds(k);
  for (size_t i = 0; i < combined_indices.size(); ++i) {
    kfolds[i % k].push_back(combined_indices[i]);
  }
  return kfolds;
}

template <typename T>
std::vector<T> index(std::vector<T> &vec, std::vector<size_t> &in) {
  std::vector<T> subset(in.size());
  for (size_t i = 0; i < in.size(); ++i)
    subset[i] = vec[in[i]];
  return subset;   
}

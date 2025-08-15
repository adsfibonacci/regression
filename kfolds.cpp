#include "kfolds.h"

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
  std::vector<size_t> fold_sizes(k);
  for (size_t i = 0; i < k; ++i) 
    fold_sizes[i] = (samples / k) + ((i < samples % k) ? 1:0);
  
  std::vector<std::vector<size_t>> kfolds(k);
  size_t start = 0;
  for (size_t i = 0; i < k; ++i) {
    std::vector<size_t> fold(indices.begin() + start,
                             indices.begin() + start + fold_sizes[i]);
    kfolds[i] = fold;
    start += fold_sizes[i];
  }    
  return kfolds;
}

std::vector<std::vector<size_t>> stratified_kfolds(size_t samples, size_t k, std::vector<double>& y, unsigned int seed) {
  std::unordered_map<double, std::vector<size_t>> class_indices;
  for(size_t i = 0; i < y.size(); ++i) 
    class_indices[y[i]].push_back(i);
  std::mt19937 rng(seed);
  std::vector<std::vector<size_t>> kfolds(k);
    
  for(auto& it : class_indices) {
    std::shuffle(it.second.begin(), it.second.end(), rng);
    std::vector<size_t> fold_sizes(k);
    for(size_t i = 0; i < k; ++i)
      fold_sizes[i] = (samples / k) + ((i < samples % k) ? 1:0);
    
    size_t start = 0;
    for(size_t i = 0; i < k; ++i) {
      kfolds[i].insert(kfolds[i].end(), it.second.begin() + start, it.second.begin() + start + fold_sizes[i]);
      start += fold_sizes[i];
    }
  }
  return kfolds;
}

template<typename T>
std::vector<T> index(std::vector<T> &vec, std::vector<size_t> &in) {
  std::vector<T> subset(in.size());
  for (size_t i = 0; i < in.size(); ++i)
    subset[i] = vec[in[i]];
  return subset;   
}

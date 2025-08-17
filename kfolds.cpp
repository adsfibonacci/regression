#include "kfolds.h"

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

void train_test_split(const std::vector<std::vector<double>> &X,
                      const std::vector<double> &y,
                      std::vector<std::vector<double>> &X_train,
                      std::vector<double> &y_train,
                      std::vector<std::vector<double>> &X_test,
                      std::vector<double> &y_test, double test_fraction,
		      unsigned random_seed) {
  size_t n_samples = X.size();
  std::vector<size_t> indices(n_samples);
  for (size_t i = 0; i < n_samples; ++i) indices[i] = i;
  
  std::mt19937 rng(random_seed);
  shuffle(indices.begin(), indices.end(), rng);
  
  size_t n_test = static_cast<size_t>(test_fraction * n_samples);
  
  X_train.clear(); y_train.clear();
  X_test.clear();  y_test.clear();
  
  for (size_t i = 0; i < n_samples; ++i) {
    size_t idx = indices[i];
    if (i < n_test) {
      X_test.push_back(X[idx]);
      y_test.push_back(y[idx]);
    } else {
      X_train.push_back(X[idx]);
      y_train.push_back(y[idx]);
    }
  }
}

void train_test_split(const std::vector<std::vector<double>> &X,
		      const std::vector<double> &y,
		      std::vector<std::vector<double>> &X_train,
		      std::vector<double> &y_train,
		      std::vector<std::vector<double>> &X_test,
		      std::vector<double> &y_test,
		      std::vector<std::vector<size_t>> &fold,
		      double test_fraction) {
  X_train.clear(); X_test.clear();
  y_train.clear(); y_test.clear();
  
  size_t test_size = std::round( test_fraction * fold.size() );
  test_size += (test_size > 1) ? 1: 0;
  size_t i = 0;
  for(; i < test_size; ++i) {
    std::vector<std::vector<double>> temp_x = index(X, fold[i]);
    std::vector<double> temp_y = index(y, fold[i]);
    X_test.insert(X_test.end(), temp_x.begin(), temp_x.end());
    y_test.insert(y_test.end(), temp_y.begin(), temp_y.end());
  }
  for(; i < fold.size(); ++i) {
    std::vector<std::vector<double>> temp_x = index(X, fold[i]);
    std::vector<double> temp_y = index(y, fold[i]);
    X_train.insert(X_train.end(), temp_x.begin(), temp_x.end());
    y_train.insert(y_train.end(), temp_y.begin(), temp_y.end());
  }
}

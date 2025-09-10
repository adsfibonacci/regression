#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>

#ifndef KFOLDS_H
#define KFOLDS_H

/**
 * Add some working-dir branch info - oops
 * Return N-folds of indices to define the X_train/test and y_train/test
 * @param samples: should be the first dimension of the design matrix
 * @param k: the number of folds to make
 * @param seed: is the random state to seed the mersenne twister
 **/
std::vector<std::vector<size_t>> kfolds(size_t samples, size_t k,
                                        unsigned int seed = 42);

/**
 * Return N-folds of indices to define the X_train/test and y_train/test in a
 * @param y: response vector with all response classes
 * @param k: the number of folds to make
 * @param seed: is the random state to seed the mersenne twister
 **/
std::vector<std::vector<size_t>>
stratified_kfolds(std::vector<double> &y, size_t k, unsigned int seed = 42);

template <typename T>
std::vector<T> index(const std::vector<T> &vec, std::vector<size_t> &in) {
  std::vector<T> subset(in.size());
  for (size_t i = 0; i < in.size(); ++i)
    subset[i] = vec[in[i]];
  return subset;   
}

/**
 * @brief Splits X and y into train and test sets by the given fraction.
 * @headerfile kfolds
 * @param X: Input (n_samples x n_features) matrix
 * @param y: Input response vector (size n_samples)
 * @param X_train: Output matrix for training samples
 * @param y_train: Output vector for training labels
 * @param X_test:  Output matrix for test samples
 * @param y_test:  Output vector for test labels
 * @param strat: Decide if the train/test split should be stratified
 * @param test_fraction: Fraction of samples to go to test (e.g., 0.2 for 20%)
 * @param random_seed:   For reproducibility (default=42)
 * @return void function due to pass by reference
 **/
void train_test_split(const std::vector<std::vector<double>> &X,
                      const std::vector<double> &y,
                      std::vector<std::vector<double>> &X_train,
                      std::vector<double> &y_train,
                      std::vector<std::vector<double>> &X_test,
                      std::vector<double> &y_test, double test_fraction = 0.2,
                      unsigned random_seed = 42);

/**
 * @brief Splits X and y into train and test sets by the given fraction and the existing kfolds
 * @headerfile kfolds
 * @param X: Input (n_samples x n_features) matrix
 * @param y: Input response vector (size n_samples)
 * @param X_train: Output matrix for training samples
 * @param y_train: Output vector for training labels
 * @param X_test:  Output matrix for test samples
 * @param y_test:  Output vector for test labels
 * @param strat: Decide if the train/test split should be stratified
 * @param test_fraction: Fraction of samples to go to test (e.g., 0.2 for 20%)
 * @param fold: The 2D list of existing kfolds, with rows listing indices in one fold
 * @return void function due to pass by reference
 **/
void train_test_split(
    const std::vector<std::vector<double>> &X, const std::vector<double> &y,
    std::vector<std::vector<double>> &X_train, std::vector<double> &y_train,
    std::vector<std::vector<double>> &X_test, std::vector<double> &y_test,
    std::vector<std::vector<size_t>> &fold, double test_fraction = 0.2);

void kminusone(
    const std::vector<std::vector<double>> &X, const std::vector<double> &y,
    std::vector<std::vector<double>> &X_train, std::vector<double> &y_train,
    std::vector<std::vector<double>> &X_test, std::vector<double> &y_test,
    std::vector<std::vector<size_t>> &fold, size_t out);


#endif

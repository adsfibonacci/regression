#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>

#ifndef KFOLDS_H
#define KFOLDS_H

std::vector<std::vector<size_t>> kfolds(size_t samples, size_t k,
                                        unsigned int seed = 42);

std::vector<std::vector<size_t>>
stratified_kfolds(std::vector<double> &y, size_t k, unsigned int seed = 42);

template <typename T>
std::vector<T> index(std::vector<T> &X, std::vector<size_t> &in);

#endif

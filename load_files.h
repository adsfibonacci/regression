#ifndef LOADER_H
#define LOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

/**
 * Returns design (n x m) matrix of an experiment with rows as samples and
 *columns as features
 * @param filename: filepath of the design matrix data file
 **/
std::vector<std::vector<double>> loadMatrix(const std::string &filename, bool norm = 1);

/**
 * Returns response vector of experiment, dimension should be n dimensional
 * @param filename: filepath of the response vector data file
 **/
std::vector<double> loadVector(const std::string &filename);

void normalize(std::vector<std::vector<double>>& X);
#endif

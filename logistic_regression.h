#ifndef LOGREG_H
#define LOGREG_H

#include <vector>
#include <cmath>
#include <string>
#include <algorithm> 
#include <ctime>

enum class Penalty { None, L1, L2 };
Penalty string_to_penalty(const std::string &s);
double sigmoid(double z);

/**
 * Trains logistic regression with optional L1/L2/no penalty using batch gradient descent.
 * @param X: n x d feature matrix (vector of n samples, each with d features)
 * @param y: size n vector with binary labels (0 or 1)
 * @param w: output vector for weights (size d)
 * @param b: output bias term
 * @param lambda: regularization strength (default 1.0)
 * @param lr: learning rate (default 0.01)
 * @param epochs: number of gradient steps (default 1000)
 * @param penalty: enum to choose L1, L2, or None penalty
 */
void logistic_regression(const std::vector<std::vector<double>> &X,
                         const std::vector<double> &y, std::vector<double> &w,
                         double &b, double lambda = 1.0, double lr = 0.01,
                         int epochs = 1000, Penalty penalty = Penalty::L2);

double predict_one(const std::vector<double> &x, const std::vector<double> &w,
                   double b);
std::vector<double>
predict_proba(const std::vector<std::vector<double>> &X_test,
              const std::vector<double> &w, double b);
std::vector<double>
predict_class(const std::vector<std::vector<double>> &X_test,
              const std::vector<double> &w, double b, double thresh = 0.5);


#endif

#ifndef LOGREG_H
#define LOGREG_H

#include "regressor.h"

class LogReg : public Regressor {
public:
  LogReg(std::vector<std::vector<double>> &X, std::vector<double> &y,
         std::vector<double> &lambdas, Penalty penalty)
    : Regressor(X, y, lambdas, penalty) {};
  
  virtual void fit(double lr = 0.05, int epochs = 1000) override;
};

class LogRegCV {
private:
  LogReg m_reg;
  std::vector<double> m_logspace;  
public:
  LogRegCV(std::vector<std::vector<double>> &X, std::vector<double> &y,
           std::vector<double> &lambdas, Penalty penalty)
    : m_reg(X, y, lambdas, penalty) {}
  void set_logspace(std::pair<double, double> region, size_t k);
};

/**
 * Trains logistic regression with optional L1/L2/no penalty using batch gradient descent.
 * @param X: n x d feature matrix (vector of n samples, each with d features)
 * @param y: size n vector with binary labels (0 or 1)
 * @param w: output vector for weights (size d)
 * @param lambda: regularization strength (default 1.0)
 * @param lr: learning rate (default 0.01)
 * @param epochs: number of gradient steps (default 1000)
 * @param penalty: enum to choose L1, L2, or None penalty
 */
void logistic_regression(const std::vector<std::vector<double>> &X,
                         const std::vector<double> &y, std::vector<double> &w,
                         double lambda = 1.0, double lr = 0.01,
                         int epochs = 1000, Penalty penalty = Penalty::L2);

/**
 * Returns the prediction of one sample via the formula <x,w> + b
 * @param x: A row/sample from the the design matrix
 * @param w: The response vector obtained from the logistic regression
 **/
double predict_one(const std::vector<double> &x, const std::vector<double> &w);

/**
 * Returns the predicted value vector for all samples
 * @param X_test: The test data set to find the predicted values on
 * @param w: The reponse vector obtained from the logistic regression
 **/
std::vector<double>
predict_proba(const std::vector<std::vector<double>> &X_test,
              const std::vector<double> &w);

/**
 * Returns the predicted classes vector for all samples
 * @param X_test: The test data set to find the predicted values on
 * @param w: The reponse vector obtained from the logistic regression
 * @param thresh: The cutoff probability between binary classication probabilities
 **/
std::vector<double>
predict_class(const std::vector<std::vector<double>> &X_test,
              const std::vector<double> &w, double thresh = 0.5);

#endif

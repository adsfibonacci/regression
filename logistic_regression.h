#ifndef LOGREG_H
#define LOGREG_H

#include "regressor.h"

class LogReg : public Regressor {
public:
  LogReg(const std::vector<std::vector<double>> &X, const std::vector<double> &y,
         double lambda, Penalty penalty)
    : Regressor(X, y, {lambda}, penalty) {};

  virtual void fit(double lr = 0.05, int epochs = 1000) override;
  void set_lambda(double l);
};

class LassoReg : public Regressor {
private:
  void _moderate_d(double lr = 0.05, int epochs = 1000);
  void _large_d(double lr = 0.05, int epochs = 1000);  
public:
  LassoReg(std::vector<std::vector<double>> &X, const std::vector<double> &y,
           double lambda)
    : Regressor(X, y, {lambda}, Penalty::L1) {};
  virtual void fit(double lr = 0.05, int epochs = 1000) override;
  void set_lambda(double l);  
};

class CV {
private:
  std::unique_ptr<Regressor> m_reg;
  std::vector<std::vector<double>> m_X;
  std::vector<double> m_y;  
  std::vector<double> m_logspace;
  std::vector<std::vector<size_t>> m_kfolds;
public:
  CV(std::vector<std::vector<double>> &X, std::vector<double> &y,
     Penalty penalty, std::vector<double> &lambdas)
    : m_reg(std::make_unique<LogReg>(X, y, lambdas[0], penalty)), m_X(X),
        m_y(y) {}  
  void set_logspace(std::pair<double, double> region, size_t k);
  double fit(size_t folds = 10, double lr = 0.05, int epochs=1000, bool strat = false, unsigned int seed = 432);
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
/**
 * Returns the prediction of one sample via the formula <x,w> + b
 * @param x: A row/sample from the the design matrix
 * @param w: The response vector obtained from the logistic regression
 **/

/**
 * Returns the predicted value vector for all samples
 * @param X_test: The test data set to find the predicted values on
 * @param w: The reponse vector obtained from the logistic regression
 **/

/**
 * Returns the predicted classes vector for all samples
 * @param X_test: The test data set to find the predicted values on
 * @param w: The reponse vector obtained from the logistic regression
 * @param thresh: The cutoff probability between binary classication probabilities
 **/

#endif

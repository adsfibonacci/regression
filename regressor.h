#ifndef REGRESSOR_H
#define REGRESSOR_H

#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>

#include "scoring.h"

enum class Penalty { None, L1, L2 };
double sigmoid(double z);

class Regressor {
private:
  double _predict(const std::vector<double> &x);
    
protected:
  std::vector<std::vector<double>> m_X;  
  std::vector<double> m_y;
  std::vector<double> m_lambdas;
  std::vector<double> m_weights;
  Penalty m_penalty;

public:
  Regressor(std::vector<std::vector<double>> &X, std::vector<double> &y,
            std::vector<double> &lambdas, Penalty penalty = Penalty::L2)
      : m_X(X), m_y(y), m_lambdas(lambdas), m_weights(X[0].size() + 1, 0.0),
        m_penalty(penalty) {    
    if (lambdas.size() == 0)
      m_lambdas = {1};
 }; 
  
  virtual void fit(double lr = 0.05, int epochs = 1000) = 0;
  std::vector<double> predict(const std::vector<std::vector<double>> &X_test);
};

#endif 

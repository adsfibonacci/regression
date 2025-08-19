#ifndef REGRESSOR_H
#define REGRESSOR_H

#include <vector>

enum class Penalty { None, L1, L2 };

class Regressor {
protected:
  std::vector<std::vector<double>> m_X;  
  std::vector<double> m_y;
  std::vector<double> m_lambdas;
  std::vector<double> m_weights;
  Penalty m_penalty;

public:
  Regressor(std::vector<std::vector<double>> &X, std::vector<double> &y,
            std::vector<double> &lambdas, Penalty penalty = Penalty::L2)
    : m_X(X), m_y(y), m_lambdas(lambdas), m_penalty(penalty) {    
    if (lambdas.size() == 0)
      lambdas = {1};
    m_weights.assign(X.size() + 1, 0.0);
    };
  
  virtual void fit(double lr=0.05, int epochs=1000) = 0;
};

#endif 

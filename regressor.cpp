#include "regressor.h"

double Regressor::_predict(const std::vector<double> &x) {
  double z = m_weights.back();
  assert(x.size() + 1 == m_weights.size());  
  for (size_t j = 0; j < m_weights.size() - 1; ++j){
    z += m_weights[j] * x[j];
  }  
  return sigmoid(z);
}
std::vector<double>
Regressor::predict(const std::vector<std::vector<double>> &X_test) {
  std::vector<double> probas;
  for (size_t i = 0; i < X_test.size(); ++i) {
    probas.push_back(_predict(X_test[i]));
  }    
  return probas;      
}      

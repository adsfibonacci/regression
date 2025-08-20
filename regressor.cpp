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
std::vector<double>
Regressor::predict_class(const std::vector<std::vector<double>> &X_test) {
  std::vector<double> probas = predict(X_test);
  std::for_each(probas.begin(), probas.end(),
                [](double &n) { n = std::round(n); });
  return probas;
}  
void Regressor::set_new(const std::vector<std::vector<double>> &X,
                        const std::vector<double> &y,
                        const std::vector<double> lambdas) {  
  m_X = X;
  m_y = y;
  m_lambdas = lambdas;  
}

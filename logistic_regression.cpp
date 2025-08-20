#include "logistic_regression.h"

void LogReg::fit(double lr, int epochs) {
  std::fill(m_weights.begin(), m_weights.end(), 0);

  size_t n = m_X.size();
  size_t d = m_X[0].size();
  std::vector<double> grad_w(d + 1, 0.0);

  for (int e = 0; e < epochs; ++e) {

    std::fill(grad_w.begin(), grad_w.end(), 0.0);

    // ----- Compute gradients -----
    for (size_t i = 0; i < n; ++i) {
      // Calculate linear combination for sample i
      double z = m_weights.back();
      for (size_t j = 0; j < d; ++j)
	z += m_weights[j] * m_X[i][j];
      double y_pred = sigmoid(z);           // Model prediction
      double error = y_pred - m_y[i];         // Error term

      // Contribute to gradients for weights and bias
      for (size_t j = 0; j < d; ++j)
	grad_w[j] += error * m_X[i][j];
      grad_w.back() += error;
    }


    for (size_t j = 0; j < d; ++j) {
      
      grad_w[j] /= n; // Normalize by number of samples

      // Add penalty gradient depending on chosen penalty

      if (m_penalty == Penalty::L2) {
        grad_w[j] +=
            m_lambdas[0] * m_weights[j]; // L2: derivative is lambda * w_j
      } else if (m_penalty == Penalty::L1) {
	// L1: subgradient is lambda * sign(w_j), not differentiable at zero
        grad_w[j] += m_lambdas[0] *
                     (m_weights[j] == 0 ? 0 : (m_weights[j] > 0 ? 1 : -1));        
      }
      // else: No penalty
      m_weights[j] -= lr * grad_w[j]; // Gradient descent step for weight
    }

    grad_w.back() /= n; // Normalize bias gradient
    m_weights.back() -= lr * grad_w.back(); // Gradient descent step for bias
  }
}
void LogReg::set_lambda(double l) {
  m_lambdas[0] = l;
}
void CV::set_logspace(std::pair<double, double> region, size_t k) {
  if (k <= 1)
    m_logspace = {std::pow(10.0, region.first)};
  double step = (region.second - region.first) / (k - 1);
  m_logspace.assign(k, 0.0);
  for (size_t i = 0; i < k; ++i)
    m_logspace[i] = std::pow(10.0, region.first + i * step);  
}
double CV::fit(size_t folds, double lr, int epochs, bool strat,
                   unsigned int seed) {
  std::vector<std::vector<double>> X_train, X_test;
  std::vector<double> y_train, y_test;
  std::vector<double> s;
  s.reserve(m_logspace.size());
  
  if (strat)
    m_kfolds = stratified_kfolds(m_y, folds, seed);
  else
    m_kfolds = kfolds(m_y.size(), folds, seed);
  for (size_t i = 0; i < m_logspace.size(); ++i) {
    kminusone(m_X, m_y, X_train, y_train, X_test, y_test, m_kfolds,
              seed % m_kfolds.size());
    m_reg->set_new(X_train, y_train);
    m_reg->fit();
    std::vector<double> y_pred = m_reg->predict(X_test);
    s.push_back(accuracy(y_test, y_pred));
  }
  
  return m_logspace[distance(s.begin(), max_element(s.begin(), s.end()))];
}

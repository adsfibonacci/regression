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
void LogReg::set_lambda(double l) { m_lambdas[0] = l; }

void LassoReg::_moderate_d(double lr, int epochs) {
  std::fill(m_weights.begin(), m_weights.end(), 0.0);
  size_t n = m_X.size();
  size_t d = m_X[0].size();
  double lambda = m_lambdas[0];
  std::vector<double> grad_w(d + 1, 0.0);
  
  for (int e = 0; e < epochs; ++e) {
    std::fill(grad_w.begin(), grad_w.end(), 0.0);
    
    // Compute full gradient
    for (size_t i = 0; i < n; ++i) {
      double pred = m_weights.back();
      for (size_t j = 0; j < d; ++j)
	pred += m_weights[j] * m_X[i][j];
      double error = pred - m_y[i];
      for (size_t j = 0; j < d; ++j)
	grad_w[j] += error * m_X[i][j];
      grad_w.back() += error; // bias
    }
    
    // Update weights (L1 subgradient)
    for (size_t j = 0; j < d; ++j) {
      grad_w[j] /= n;
      grad_w[j] += lambda * (m_weights[j] == 0 ? 0 : (m_weights[j] > 0 ? 1 : -1));
      m_weights[j] -= lr * grad_w[j];
    }
    grad_w.back() /= n;
    m_weights.back() -= lr * grad_w.back();
  }
}
void LassoReg::_large_d(double lr, int epochs) {
  size_t n = m_X.size(), d = m_X[0].size();
  // for (size_t i = 0; i <=d; ++i) std::cout << m_weights[i] << std::endl;
  
  assert((n > 0) && (m_weights.size() == d + 1) && (d > 0));
  
  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::vector<double> grad(d + 1, 0.0);
    
    for (size_t i = 0; i < n; ++i) {
      // Prediction: w0 + X_i dot w_rest
      double y_pred = m_weights.back();
      assert(d == m_X[i].size());
      for (size_t j = 0; j < d; ++j)
	y_pred += m_X[i][j] * m_weights[j];
      
      double error = y_pred - m_y[i];
      // Gradient for bias
      grad.back() += error;
      
      // Gradient for weights
      for (size_t j = 0; j < d; ++j)
	grad[j] += error * m_X[i][j];
    }
    // Average gradient
    for (size_t j = 0; j <= d; ++j)
      grad[j] /= n;
    
    // Update weights
    m_weights.back() -= lr * grad.back(); // bias (no regularization)
    for (size_t j = 0; j < d; ++j) {
      // Gradient + L1 penalty (subgradient)
      double subgrad = grad[j] + m_lambdas[0] * ((m_weights[j] > 0) - (m_weights[j] < 0));
      m_weights[j] -= lr * subgrad;
    }
  }
  // for (size_t i = 0; i <=d; ++i) std::cout << m_weights[i] << std::endl;
}
void LassoReg::fit(double lr, int epochs) {
  std::fill(m_weights.begin(), m_weights.end(), 0.0);
  if (m_X[0].size() > m_X.size() * 2)
    _large_d(lr, epochs);
  else
    _moderate_d(lr,epochs);    
}
void LassoReg::set_lambda(double l) { m_lambdas[0] = l; }

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
              (int)time(0) % m_kfolds.size());
    m_reg->set_new(X_train, y_train, {m_logspace[i]});
    m_reg->fit();
    std::vector<double> y_pred = m_reg->predict(X_test);
    s.push_back(accuracy(y_test, y_pred));
  }
  for (auto x : s) {std::cout<<x<<std::endl;}
  return m_logspace[distance(s.begin(), max_element(s.begin(), s.end()))];
}

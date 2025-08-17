#include "logistic_regression.h"

Penalty string_to_penalty(const std::string& s) {
  if (s == "l1") return Penalty::L1;
  if (s == "l2") return Penalty::L2;
  return Penalty::None;
}
double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

void logistic_regression(const std::vector<std::vector<double>> &X,
                         const std::vector<double> &y, std::vector<double> &w,
			 double lambda, double lr,
                         int epochs, Penalty penalty) {
  
  size_t n = X.size();     // Number of training examples
  size_t d = X[0].size();  // Number of features
  w.assign(d+1, 0.0);        // Initialize weights to zero
  
  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::vector<double> grad_w(d+1, 0.0); // Gradient accumulator for weights
    
    // ----- Compute gradients -----
    for (size_t i = 0; i < n; ++i) {
      // Calculate linear combination for sample i
      double z = w.back();
      for (size_t j = 0; j < d; ++j)
	z += w[j] * X[i][j];
      double y_pred = sigmoid(z);           // Model prediction
      double error = y_pred - y[i];         // Error term
      
      // Contribute to gradients for weights and bias
      for (size_t j = 0; j < d; ++j)
	grad_w[j] += error * X[i][j];
      grad_w.back() += error;
    }
    
    // ----- Normalize and add penalty gradients -----
    for (size_t j = 0; j < d; ++j) {
      grad_w[j] /= n; // Normalize by number of samples
      
      // Add penalty gradient depending on chosen penalty
      if (penalty == Penalty::L2) {
	grad_w[j] += lambda * w[j]; // L2: derivative is lambda * w_j
      } else if (penalty == Penalty::L1) {
	// L1: subgradient is lambda * sign(w_j), not differentiable at zero
	grad_w[j] += lambda * (w[j] == 0 ? 0 : (w[j] > 0 ? 1 : -1));
      }
      // else: No penalty
      w[j] -= lr * grad_w[j]; // Gradient descent step for weight
    }
    
    grad_w.back() /= n;                 // Normalize bias gradient
    w.back() -= lr * grad_w.back();            // Gradient descent step for bias
    
    // ----- (Optional) Print loss every 100 epochs -----
    if ((epoch+1) % 100 == 0) {
      double loss = 0.0;
      for (size_t i = 0; i < n; ++i) {
	double z = w.back();
	for (size_t j = 0; j < d; ++j)
	  z += w[j] * X[i][j];
	double p = sigmoid(z);
	
	// Clamp to avoid log(0)
	p = std::clamp(p, 1e-12, 1-1e-12);
	
	// Logistic loss for this example
	loss += -y[i]*std::log(p) - (1-y[i])*std::log(1-p);
      }
      loss /= n; // Mean loss
      
      // Add penalty to loss for monitoring
      double reg = 0.0;
      if (penalty == Penalty::L2)
	for (double wi : w) reg += 0.5 * lambda * wi * wi;   // Ridge penalty
      else if (penalty == Penalty::L1)
	for (double wi : w) reg += lambda * std::abs(wi);   // Lasso penalty
      loss += reg;
      
      // cout << "Epoch " << epoch+1 << " Loss = " << loss << endl;
    }
  }
}
// Returns sigmoid(w^T x + b)
double predict_one(const std::vector<double> &x,
                   const std::vector<double> &w) {
  double z = w.back();
  for (size_t j = 0; j < w.size() - 1; ++j)
    z += w[j] * x[j];
  return 1.0 / (1.0 + std::exp(-z));
}

// Predicts probabilities for all rows in X_test
std::vector<double>
predict_proba(const std::vector<std::vector<double>> &X_test,
              const std::vector<double> &w) {
  std::vector<double> probas;
  for (const auto& x : X_test)
    probas.push_back(predict_one(x, w));
  return probas;
}
// Predicts classes (0/1) for all test samples using a threshold (default 0.5)
std::vector<double>
predict_class(const std::vector<std::vector<double>> &X_test,
              const std::vector<double> &w, double thresh) {
  std::vector<double> preds;
  for (const auto& x : X_test) {
    double p = predict_one(x, w);
    preds.push_back(p >= thresh ? 1 : 0);
  }
  return preds;
}


// Example usage: train on AND logic
// int main() {
//   vector<vector<double>> X = loadMatrix("design_matrix.txt");
//   vector<double> y = loadVector("responses.txt");
//   vector<vector<double>> X_train, X_test;
//   vector<double> y_train, y_test;
//     train_test_split(X, y, X_train, y_train, X_test, y_test, .333, (int)time(0));
//     
//     // Output parameters
//     vector<double> w;
//     double b;
//     
//     // Choose penalty: "l2", "l1", or "none"
//     string method = "l2";
//     logistic_regression(X_train, y_train, w, b, 1.0, 0.1, 1000,
// 			string_to_penalty(method));
//     vector<double> y_pred = predict_class(X_test, w, b, .5);
//     
//     cout << "Accuracy: " << accuracy(y_test, y_pred) << endl;  
//     cout << "R2: " << r2_score(y_test, y_pred) << endl;
//     cout << "Learned bias: " << b << "\n";
// }

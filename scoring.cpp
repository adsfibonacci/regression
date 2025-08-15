#include <numeric>
#include <vector>
#include <limits>

/**
 * Returns the fraction of equal entries in two vectors as accuracy.
 * Both vectors must be of same length and contain values 0 or 1.
 */
double accuracy(const std::vector<double> &y_true,
                const std::vector<double> &y_pred) {
  if (y_pred.size() != y_true.size() || y_pred.empty())
    return 0.0;
  int correct = 0;
  for (size_t i = 0; i < y_pred.size(); ++i)
    if (y_pred[i] == y_true[i])
      ++correct;
  return double(correct) / y_pred.size();
}
/**
 * Computes the Residual Sum of Squares (RSS)
 * y_true: true target values (e.g., labels or regression targets)
 * y_pred: model predicted values (e.g., probabilities for logistic regression)
 */
double rss(const std::vector<double> &y_true,
           const std::vector<double> &y_pred) {
  if (y_true.size() != y_pred.size() || y_true.empty())
    return std::numeric_limits<double>::infinity();
  double rss = 0.0;
  for (size_t i = 0; i < y_true.size(); ++i)
    rss += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
  return rss;
}
/**
 * Computes the Total Sum of Squares between predictions and true targets.
 * y_true: true target values (e.g., labels or regression targets)
 * y_pred: model predicted values (e.g., probabilities for logistic regression)
 */
double tss(const std::vector<double> &y_true,
           const std::vector<double> &y_pred) {
  if (y_true.size() != y_pred.size() || y_true.empty())
    return std::numeric_limits<double>::infinity();
  double mean = std::accumulate(y_true.begin(), y_true.end(), 0.0);
  double tss = 0.0;
  for (size_t i = 0; i < y_true.size(); ++i)
    tss += (y_true[i] - mean) * (y_true[i] - mean);
  return tss;
}
/**
 * Computes the R^2 (coefficient of determination) between predictions and true targets.
 * y_true: true target values (e.g., labels or regression targets)
 * y_pred: model predicted values (e.g., probabilities for logistic regression)
 */
double r2_score(const std::vector<double> &y_true,
                const std::vector<double> &y_pred) {
  if (y_true.size() != y_pred.size() || y_true.empty())
    return std::numeric_limits<double>::infinity();  
  double ss_res = rss(y_true, y_pred), ss_tot = tss(y_true, y_pred); 
  
  if (ss_tot == 0.0) return 0.0;
  
  return 1.0 - ss_res / ss_tot;
}

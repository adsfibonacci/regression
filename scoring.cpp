#include "scoring.h"


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

double rss(const std::vector<double> &y_true,
           const std::vector<double> &y_pred) {
  if (y_true.size() != y_pred.size() || y_true.empty())
    return std::numeric_limits<double>::infinity();
  double rss = 0.0;
  for (size_t i = 0; i < y_true.size(); ++i)
    rss += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
  return rss;
}

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

double r2_score(const std::vector<double> &y_true,
                const std::vector<double> &y_pred) {
  if (y_true.size() != y_pred.size() || y_true.empty())
    return std::numeric_limits<double>::infinity();  
  double ss_res = rss(y_true, y_pred), ss_tot = tss(y_true, y_pred); 
  
  if (ss_tot == 0.0) return 0.0;
  
  return 1.0 - ss_res / ss_tot;
}

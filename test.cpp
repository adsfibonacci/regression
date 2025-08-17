#include <iostream>
#include <ctime>
#include "load_files.h"
#include "kfolds.h"
#include "logistic_regression.h"
#include "scoring.cpp"

using namespace std;

template <typename T> ostream &operator<<(ostream &os, const vector<T> &v) {
  for (size_t i = 0; i < v.size(); ++i)
    os << v[i] << "\t";
  return os;
}
template <typename T> ostream &operator<<(ostream &os, const vector<vector<T>> &v) {  
  for (size_t i = 0; i < v.size(); ++i)
    os << v[i] << "\n";
  return os;
}

int main() {
  vector<vector<double>> X = loadMatrix("design_matrix.txt");
  vector<double> y = loadVector("responses.txt");
  vector<vector<double>> X_train, X_test;
  vector<double> y_train, y_test;
  train_test_split(X, y, X_train, y_train, X_test, y_test, .333, (int)time(0));
  vector<double> w;
  double b;
  logistic_regression(X_train, y_train, w, b);
  cout << "Accuracy: " << accuracy(y_test, predict_class(X_test, w, b, .5)) << endl;
  cout << "R2: " << r2_score(y_test, predict_class(X_test, w, b, .5)) << endl;
}  

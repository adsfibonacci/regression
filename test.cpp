#include <iostream>
#include <ctime>
#include "load_files.h"
#include "kfolds.h"
#include "logistic_regression.h"
#include "scoring.h"

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
  vector<vector<size_t>> folds = kfolds(y.size(), 10, (int)time(0));
  train_test_split(X, y, X_train, y_train, X_test, y_test, folds, .2);

  cout << X_test.size() << endl;
  cout << X_train.size() << endl;
  
  vector<double> w;
  logistic_regression(X_train, y_train, w, 1, .05, 500);
  cout << "Accuracy: " << accuracy(y_test, predict_class(X_test, w, .5)) << endl;
  cout << "R2: " << r2_score(y_test, predict_class(X_test, w, .5)) << endl;
}  

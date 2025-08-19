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

int main() {
  vector<vector<double>> X = loadMatrix("design_matrix.txt");
  vector<double> y = loadVector("responses.txt");
  
  vector<vector<double>> X_train, X_test;
  vector<double> y_train, y_test;
  vector<vector<size_t>> folds = kfolds(y.size(), 10, (int)time(0));
  train_test_split(X, y, X_train, y_train, X_test, y_test, folds, .2);

  cout << X_test.size() << endl;
  cout << X_train.size() << endl;

  std::vector<double> l;
  LogReg reg(X_train, y_train, l, Penalty::L2);
  reg.fit();

  std::vector<double> y_pred = reg.predict(X_test);
  cout << "Accuracy: " << accuracy(y_test, y_pred) << endl;
  cout << "R2: " << r2_score(y_test, y_pred) << endl;
  return 0;
}  

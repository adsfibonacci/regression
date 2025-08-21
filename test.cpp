#include <iostream>
#include <ctime>
#include "load_files.h"
#include "kfolds.h"
#include "logistic_regression.h"
// #include "scoring.h"

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
  vector<vector<size_t>> folds = stratified_kfolds(y, 3, (int)time(0));
  kminusone(X, y, X_train, y_train, X_test, y_test, folds, (int)time(0) % 3);
  
  cout << "Training size: " << X_train.size() << endl;
  cout << "Testing size: " << X_test.size() << endl;

  LassoReg mod(X_train, y_train, 1.0);
  mod.fit(0.00001);
  
  vector<double> y_pred = mod.predict_class(X_test);
  cout << y_pred << endl;
  cout << "Accuracy: " << accuracy(y_test, y_pred) << endl;
  cout << "R2: " << r2_score(y_test, y_pred) << endl;
  return 0;
} 

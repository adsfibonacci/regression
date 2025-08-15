#include <iostream>
#include <ctime>
#include "load_files.h"
#include "kfolds.h"

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
  vector<vector<size_t>> folds = stratified_kfolds(y, 10);
  cout << folds << endl;  
}  

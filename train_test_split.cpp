#include <algorithm>
#include <random>
#include <vector>

using std::vector, std::shuffle, std::mt19937;

/**
 * Splits X and y into train and test sets by the given fraction.
 * @param X: Input (n_samples x n_features) matrix
 * @param y: Input response vector (size n_samples)
 * @param X_train: Output matrix for training samples
 * @param y_train: Output vector for training labels
 * @param X_test:  Output matrix for test samples
 * @param y_test:  Output vector for test labels
 * @param test_fraction: Fraction of samples to go to test (e.g., 0.2 for 20%)
 * @param random_seed:   For reproducibility (default=42)
 */
void train_test_split(
    const vector<vector<double>>& X,
    const vector<double>& y,
    vector<vector<double>>& X_train,
    vector<double>& y_train,
    vector<vector<double>>& X_test,
    vector<double>& y_test,
    double test_fraction = 0.2,
    unsigned random_seed = 42)
{
    size_t n_samples = X.size();
    vector<size_t> indices(n_samples);
    for (size_t i = 0; i < n_samples; ++i) indices[i] = i;

    // Fisher-Yates shuffle for randomness
    mt19937 rng(random_seed);
    shuffle(indices.begin(), indices.end(), rng);

    size_t n_test = static_cast<size_t>(test_fraction * n_samples);

    X_train.clear(); y_train.clear();
    X_test.clear();  y_test.clear();

    for (size_t i = 0; i < n_samples; ++i) {
        size_t idx = indices[i];
        if (i < n_test) {
            X_test.push_back(X[idx]);
            y_test.push_back(y[idx]);
        } else {
            X_train.push_back(X[idx]);
            y_train.push_back(y[idx]);
        }
    }
}

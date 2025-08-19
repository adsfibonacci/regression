#ifndef SCORING_H
#define SCORING_H

#include <cmath>
#include <vector>
#include <limits>
#include <numeric>

/**
 * @brief Returns the fraction of equal entries in two vectors as accuracy.
 * @param y_true: The response values from the test data
 * @param y_pred: The expected values for the test data
 * Both vectors must be of same length and contain values 0 or 1.
 **/
double accuracy(const std::vector<double> &y_true,
                const std::vector<double> &y_pred);

/**
 * @brief Returns the Residual Sum of Squares (RSS)
 * @param y_true: The response values from the test data
 * @param y_pred: The expected values for the test data
 * Both vectors must be of same length and contain values 0 or 1.
 **/
double rss(const std::vector<double> &y_true,
           const std::vector<double> &y_pred);

/**
 * @brief Returns the Total Sum of Squares (TSS)
 * @param y_true: The response values from the test data
 * @param y_pred: The expected values for the test data
 * Both vectors must be of same length and contain values 0 or 1.
 **/
double tss(const std::vector<double> &y_true,
           const std::vector<double> &y_pred);

/**
 * @brief Computes the R^2 (coefficient of determination) between predictions and true targets
 * @param y_true: The response values from the test data
 * @param y_pred: The expected values for the test data
 * Both vectors must be of same length and contain values 0 or 1.
 **/
double r2_score(const std::vector<double> &y_true,
                const std::vector<double> &y_pred);

double sigmoid(double z);
#endif

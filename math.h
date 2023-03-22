#include <bits/stdc++.h>
#define double long double
using namespace std;

// Random double
double rand_double(double min_val, double max_val) {
    double f = (double)rand() / RAND_MAX;
    return min_val + f * (max_val - min_val);
}

// Index of maximum in array
int argmax(vector<double> arr) {
    double mx = arr[0];
    int mx_ind = 0;
    for (int i = 0; i < arr.size(); ++i) {
        if(mx < arr[i]) {
            mx = arr[i];
            mx_ind = i;
        }
    }
    return mx_ind;
}

// Transpose 2D array of double
vector<vector<double>> transparent(vector<vector<double>>& arr) {
    vector<vector<double>> transposed(arr[0].size(), vector<double> (arr.size()));
    for (int i = 0; i < arr.size(); ++i) {
        for (int j = 0; j < arr[0].size(); ++j) {
            transposed[j][i] = arr[i][j];
        }
    }
    return transposed;
}

double relu(double x) {
    return (x >= 0) * x;
}

double relu2deriv(double output) {
    return output >= 0;
}

double tanh2deriv(double x) {
    return 1 - (tanh(x) * tanh(x));
}

vector<double> softmax(const vector<double>& input) {
    vector<double> output(input.size());
    double sum = 0.0;
    for (int i = 0; i < input.size(); i++) {
        output[i] = exp(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < output.size(); i++) {
        output[i] /= sum;
    }
    return output;
}

#include <bits/stdc++.h>
#include "perceptron.h"
#define double long double
using namespace std;
const string TRAIN_PATH = "D:\\LettersDataset\\Train\\Letters";
const string TRAIN_LABEL_PATH = "D:\\LettersDataset\\Train\\Labels";
const string TEST_PATH = "D:\\LettersDataset\\Test\\Letters";
const string TEST_LABEL_PATH = "D:\\LettersDataset\\Test\\Labels";
vector <vector <vector <int>>> letters_train;
vector <vector <vector <int>>> letters_test;
vector <int> labels_train;
vector <int> labels_test;
const int TRAIN_SIZE = 60;
const int TEST_SIZE = 6;

void load_train_data() {
    letters_train.resize(TRAIN_SIZE, vector <vector <int>> (28, vector <int> (28)));
    labels_train.resize(TRAIN_SIZE);
    // Loading letters train data
    for(int cnt = 0; cnt < TRAIN_SIZE; ++cnt) {
        ifstream in;
        in.open(TRAIN_PATH + "\\" + "letter" + to_string(cnt + 1) + ".txt", ios::in);
        for(int i = 0; i < 28; ++i) {
            for(int j = 0; j < 28; ++j) {
                int x; in >> x;
                letters_train[cnt][i][j] = x;
            }
        }
    }
    for(int cnt = 0; cnt < TRAIN_SIZE; ++cnt) {
        ifstream in;
        in.open(TRAIN_LABEL_PATH + "\\" + "label" + to_string(cnt + 1) + ".txt", ios::in);
        int x;
        in >> x;
        labels_train[cnt] = x - 1;
    }
}

void load_test_data() {
    letters_test.resize(TEST_SIZE, vector <vector <int>> (28, vector <int> (28)));
    labels_test.resize(TEST_SIZE);
    // Loading letters train data
    for(int cnt = 0; cnt < TEST_SIZE; ++cnt) {
        ifstream in;
        in.open(TEST_PATH + "\\" + "letter" + to_string(cnt + 1) + ".txt", ios::in);
        for(int i = 0; i < 28; ++i) {
            for(int j = 0; j < 28; ++j) {
                int x; in >> x;
                letters_test[cnt][i][j] = x;
            }
        }
    }
}

const double alpha = 0.005;
const int iterations = 200;
const int hidden_size = 100;
const int pixels_per_image = 28 * 28;
const int num_labels = 33;

int main() {
    srand(time(nullptr));
    load_train_data();
    load_test_data();
    Perceptron neural(pixels_per_image, hidden_size, num_labels, alpha);
    auto converted_data = convert_train_data(letters_train);
    neural.train(converted_data, labels_train, iterations);
    auto converted_test_data = convert_train_data(letters_test);
    neural.custom_test(converted_test_data);
}

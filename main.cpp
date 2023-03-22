#include <bits/stdc++.h>
#include "perceptron.h"
#define double long double
using namespace std;
const string TRAIN_PATH = "D:\\LettersDataset\\Train\\Letters";
const string TRAIN_LABEL_PATH = "D:\\LettersDataset\\Train\\Labels";
const string TEST_PATH = "D:\\LettersDataset\\Test\\Letters";
const string TEST_LABEL_PATH = "D:\\LettersDataset\\Test\\Labels";
vector<vector<vector<int>>> letters_train;
vector<vector<vector<int>>> letters_test;
vector<int> labels_train;
vector<int> labels_test;
const int TRAIN_SIZE = 310;
const int TEST_SIZE = 31;

void load_train_data() {
    letters_train.resize(TRAIN_SIZE, vector<vector<int>> (28, vector<int> (28)));
    labels_train.resize(TRAIN_SIZE);
    // Loading letters train data
    for (int cnt = 0; cnt < TRAIN_SIZE; ++cnt) {
        ifstream in;
        in.open(TRAIN_PATH + "\\" + "letter" + to_string(cnt + 1) + ".txt", ios::in);
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                int x; in >> x;
                letters_train[cnt][i][j] = x;
            }
        }
    }
    for (int cnt = 0; cnt < TRAIN_SIZE; ++cnt) {
        ifstream in;
        in.open(TRAIN_LABEL_PATH + "\\" + "label" + to_string(cnt + 1) + ".txt", ios::in);
        int x;
        in >> x;
        labels_train[cnt] = x - 1;
    }
}

void load_test_data() {
    letters_test.resize(TEST_SIZE, vector<vector<int>> (28, vector<int> (28)));
    labels_test.resize(TEST_SIZE);
    // Loading letters train data
    for (int cnt = 0; cnt < TEST_SIZE; ++cnt) {
        ifstream in;
        in.open(TEST_PATH + "\\" + "letter" + to_string(cnt + 1) + ".txt", ios::in);
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                int x; in >> x;
                letters_test[cnt][i][j] = x;
            }
        }
    }
    for (int cnt = 0; cnt < TEST_SIZE; ++cnt) {
        ifstream in;
        in.open(TEST_LABEL_PATH + "\\" + "label" + to_string(cnt + 1) + ".txt", ios::in);
        int x;
        in >> x;
        labels_test[cnt] = x - 1;
    }
}

const double alpha = 0.001;
const int iterations = 300;
const int hidden_size = 100;
const int pixels_per_image = 28 * 28;
const int num_labels = 31;

int main() {
    srand(time(nullptr));
    load_train_data();
    load_test_data();
    Perceptron neural(pixels_per_image, hidden_size, num_labels, alpha);
    auto converted_data = convert_train_data(letters_train);
    auto converted_test_data = convert_train_data(letters_test);
    neural.load_weights("D:\\LettersDataset\\Weights");
    // menu
    while (true) {
        cout << "Choose function: " << '\n';
        cout << " 0) Pass test data\n";
        cout << " 1) Train network" << '\n';
        cout << " 2) Ask network custom test" << '\n';
        cout << " 3) Save" << '\n';
        cout << " 4) Manual train" << '\n';
        cout << " 5) Exit" << '\n';
        int chose;
        cin >> chose;
        if (chose < 0 || chose > 5) {
            cout << "Incorrect choice\n";
            continue;
        }
        else if (chose == 5) {
            break;
        }
        else if (chose == 1) {
            neural.train(converted_data, labels_train, converted_test_data, labels_test, iterations);
            //neural.train(letters_train, labels_train, letters_test, labels_test, iterations);
        }
        else if (chose == 2) {
            cout << "Write path to file with letter\n";
            string path;
            cin >> path;
            vector<vector<int>> letter(28, vector<int> (28));
            ifstream in;
            in.open(path, ios::in);
            if (!in.is_open()) {
                cout << "Wrong file!\n";
                continue;
            }
            for (int i = 0; i < 28; ++i) {
                for (int j = 0; j < 28; ++j) {
                    in >> letter[i][j];
                }
            }
            letter = convert_train_data({letter})[0];
            int res = neural.ask(letter);
            cout << res + 1 << '\n';
        }
        else if (chose == 3) {
            neural.save_weights("D:\\LettersDataset\\Weights");
        }
        else if (chose == 4) {
            cout << "Write path to file with letter\n";
            string path;
            cin >> path;
            cout << "Write answer:\n";
            int ans;
            cin >> ans;
            ans--;
            vector<vector<int>> letter(28, vector<int> (28));
            ifstream in;
            in.open(path, ios::in);
            if (!in.is_open()) {
                cout << "Wrong file!\n";
                continue;
            }
            for (int i = 0; i < 28; ++i) {
                for (int j = 0; j < 28; ++j) {
                    in >> letter[i][j];
                }
            }
            auto letter_train = convert_train_data({letter});
            vector<int> a = {ans};
            neural.iteration(letter_train, a);
            neural.iteration(letter_train, a);
        }
    }
}

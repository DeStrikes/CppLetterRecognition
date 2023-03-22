#include <bits/stdc++.h>
#include "math.h"
using namespace std;
#define double long double

vector<vector<vector<int>>> convert_train_data(vector<vector<vector<int>>> train_data) {
    int size = train_data.size();
    vector<vector<vector<int>>> res(size);
    for (int cnt = 0; cnt < size; ++cnt) {
        int mn_left = 100, mn_up = 100;
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                if(train_data[cnt][i][j] == 1) {
                    mn_left = min(mn_left, j);
                    mn_up = min(mn_up, i);
                }
            }
        }
        vector<vector<int>> new_letter;
        for (int i = mn_up; i < 28; ++i) {
            new_letter.push_back({});
            for (int j = mn_left; j < 28; ++j) {
                new_letter[new_letter.size() - 1].push_back(train_data[cnt][i][j]);
            }
            while(new_letter.back().size() < 28)
                new_letter.back().push_back(0);
        }
        while(new_letter.size() < 28) {
            new_letter.push_back({});
            while(new_letter.back().size() < 28)
                new_letter.back().push_back(0);
        }
        res[cnt] = new_letter;
    }
    return res;
}

void output_letter(vector<vector<int>> letter) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (letter[i][j]) {
                cout << '#';
            } else {
                cout << '.';
            }
            cout << '.';
        }
        cout << '\n';
    }
}

class Perceptron {
public:
    Perceptron() {}
    Perceptron(int input_layer, int hidden_layer, int out_layer, double alpha, bool need_init = true) {
        this->input_layer = input_layer;
        this->hidden_layer = hidden_layer;
        this->out_layer = out_layer;
        this->alpha = alpha;
    }

    void init_weights() {
        srand(time(nullptr));
        weights_0_1.resize(input_layer, vector<double> (hidden_layer));
        for (int i = 0; i < input_layer; ++i) {
            for (int j = 0; j < hidden_layer; ++j) {
                weights_0_1[i][j] = rand_double(-0.02, 0.02);
            }
        }
        weights_1_2.resize(hidden_layer, vector<double> (out_layer));
        for (int i = 0; i < hidden_layer; ++i) {
            for (int j = 0; j < out_layer; ++j) {
                weights_1_2[i][j] = rand_double(-0.02, 0.02);
            }
        }
    }

    void load_weights(string path) {
        weights_0_1.resize(input_layer, vector<double> (hidden_layer));
        weights_1_2.resize(hidden_layer, vector<double> (out_layer));
        string name_0_1 = "weights_0_1.txt";
        string path_0_1 = path + "\\" + name_0_1;
        ifstream in1;
        in1.open(path_0_1, ios::in);
        for (int i = 0; i < input_layer; ++i) {
            for (int j = 0; j < hidden_layer; ++j) {
                in1 >> weights_0_1[i][j];
            }
        }
        string name_1_2 = "weights_1_2.txt";
        string path_1_2 = path + "\\" + name_1_2;
        ifstream in2;
        in2.open(path_1_2, ios::in);
        for (int i = 0; i < hidden_layer; ++i) {
            for (int j = 0; j < out_layer; ++j) {
                in2 >> weights_1_2[i][j];
            }
        }
    }

    void save_weights(string path) {
        string name_0_1 = "weights_0_1.txt";
        string path_0_1 = path + "\\" + name_0_1;
        ofstream out1;
        out1.open(path_0_1, ios::out);
        for (int i = 0; i < weights_0_1.size(); ++i) {
            for (int j = 0; j < weights_0_1[0].size(); ++j) {
                out1 << weights_0_1[i][j] << " ";
            }
            out1 << '\n';
        }
        string name_1_2 = "weights_1_2.txt";
        string path_1_2 = path + "\\" + name_1_2;
        ofstream out2;
        out2.open(path_1_2);
        for (int i = 0; i < weights_1_2.size(); ++i) {
            for (int j = 0; j < weights_1_2[0].size(); ++j) {
                out2 << weights_1_2[i][j] << " ";
            }
            out2 << '\n';
        }
    }

    void train(vector<vector<vector<int>>>& train_data, vector<int>& labels,
               vector<vector<vector<int>>>& test_data, vector<int>& test_labels,
               int iterations, bool is_output = true, bool need_all_test = true) {
        init_weights();
        for (int it = 1; it <= iterations; ++it) {
            int count = iteration(train_data, labels);
            if(is_output) {
                cout << "#";
                if(it < 10) {
                    cout << it << "     ";
                }
                else if(it < 100) {
                    cout << it << "    ";
                }
                else if(it < 1000) {
                    cout << it << "   ";
                }
                else if(it < 10000) {
                    cout << it << "  ";
                }
                cout << "Train data passed: " << count << '\n';
            }
            if(need_all_test) {
                int correct_train = 0;
                for (int cnt = 0; cnt < train_data.size(); ++cnt) {
                    int res = ask(train_data[cnt]);
                    if (res == labels[cnt]) {
                        correct_train++;
                    }
                }
                cout << "       " << "Passed test in all train data: " << correct_train << "/" << train_data.size() << '\n';
            }
            int correct_test = 0;
            for (int i = 0; i < test_data.size(); ++i) {
                int res = ask(test_data[i]);
                if (res == test_labels[i]) {
                    correct_test++;
                }
            }
            cout << "       " << "Passed test cases: " << correct_test << "/" << test_data.size() << '\n';
        }
    }

    // ask network letter
    int ask(vector<vector<int>>& letter) {
        vector<double> layer_0(input_layer);
        // Loading test data into 1-d array
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                layer_0[i * 28 + j] = letter[i][j];
            }
        }
        vector<double> layer_1(hidden_layer, 0);
        for (int j = 0; j < hidden_layer; ++j) {
            for (int i = 0; i < input_layer; ++i) {
                layer_1[j] += layer_0[i] * weights_0_1[i][j];
            }
        }
        for (int i = 0; i < hidden_layer; ++i) {
            layer_1[i] = tanh(layer_1[i]);
        }
        vector<double> layer_2(out_layer, 0);
        for (int j = 0; j < out_layer; ++j) {
            for (int i = 0; i < hidden_layer; ++i) {
                layer_2[j] += layer_1[i] * weights_1_2[i][j];
            }
        }
        layer_2 = softmax(layer_2);
        int cur_ans = argmax(layer_2);
        return cur_ans;
    }

    int iteration(vector<vector<vector<int>>>& train_data, vector<int>& labels) {
        int correct_cnt = 0;
        int size = train_data.size();
        for (int cnt = 0; cnt < size; ++cnt) {
            vector<double> layer_0(input_layer);
            // Loading test data into 1-d array
            for (int i = 0; i < 28; ++i) {
                for (int j = 0; j < 28; ++j) {
                    layer_0[i * 28 + j] = train_data[cnt][i][j];
                }
            }
            vector<double> layer_1(hidden_layer, 0);
            for (int j = 0; j < hidden_layer; ++j) {
                for (int i = 0; i < input_layer; ++i) {
                    layer_1[j] += layer_0[i] * weights_0_1[i][j];
                }
            }
            for (int i = 0; i < hidden_layer; ++i) {
                layer_1[i] = tanh(layer_1[i]);
            }
            vector<double> dropout_mask(hidden_layer);
            for (int i = 0; i < hidden_layer; ++i)
                dropout_mask[i] = rand() % 2;
            for (int i = 0; i < hidden_layer; ++i) {
                layer_1[i] *= dropout_mask[i] * 2;
            }
            vector<double> layer_2(out_layer, 0);
            for (int j = 0; j < out_layer; ++j) {
                for (int i = 0; i < hidden_layer; ++i) {
                    layer_2[j] += layer_1[i] * weights_1_2[i][j];
                }
            }
            layer_2 = softmax(layer_2);
            int cur_ans = argmax(layer_2);
            if(cur_ans == labels[cnt])
                correct_cnt++;
            vector<double> layer_2_delta(out_layer);
            int ans_label = labels[cnt];
            for (int i = 0; i < out_layer; ++i) {
                if(i == ans_label) {
                    layer_2_delta[i] = 1 - layer_2[i];
                }
                else {
                    layer_2_delta[i] = 0 - layer_2[i];
                }
            }
            vector<double> layer_1_tanh2deriv(hidden_layer);
            for (int i = 0; i < hidden_layer; ++i)
                layer_1_tanh2deriv[i] = tanh2deriv(layer_1[i]);
            vector<vector<double>> weights_1_2_T = transparent(weights_1_2);
            vector<double> layer_1_delta(hidden_layer, 0);
            for (int j = 0; j < hidden_layer; ++j) {
                for (int i = 0; i < out_layer; ++i) {
                    layer_1_delta[j] += layer_2_delta[i] * weights_1_2_T[i][j] * layer_1_tanh2deriv[j];
                }
            }
            for (int i = 0; i < hidden_layer; ++i) {
                layer_1_delta[i] *= dropout_mask[i];
            }
            for (int i = 0; i < hidden_layer; ++i) {
                for (int j = 0; j < out_layer; ++j) {
                    weights_1_2[i][j] += layer_1[i] * layer_2_delta[j] * alpha;
                }
            }
            for (int i = 0; i < input_layer; ++i) {
                for (int j = 0; j < hidden_layer; ++j) {
                    weights_0_1[i][j] += layer_0[i] * layer_1_delta[j] * alpha;
                }
            }
        }
        return correct_cnt;
    }

private:
    double alpha;
    int input_layer, hidden_layer, out_layer;
    vector<vector<double>> weights_0_1;
    vector<vector<double>> weights_1_2;
};
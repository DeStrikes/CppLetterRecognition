#include <bits/stdc++.h>
#include "math.h"
using namespace std;
#define double long double

vector <vector <vector <int>>> convert_train_data(vector <vector <vector <int>>> train_data) {
    int size = train_data.size();
    vector <vector <vector <int>>> res(size);
    for(int cnt = 0; cnt < size; ++cnt) {
        int mn_left = 100, mn_up = 100;
        for(int i = 0; i < 28; ++i) {
            for(int j = 0; j < 28; ++j) {
                if(train_data[cnt][i][j] == 1) {
                    mn_left = min(mn_left, j);
                    mn_up = min(mn_up, i);
                }
            }
        }
        vector <vector <int>> new_letter;
        for(int i = mn_up; i < 28; ++i) {
            new_letter.push_back({});
            for(int j = mn_left; j < 28; ++j) {
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

void output_letter(vector <vector <int>> letter) {
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
    Perceptron(int input_layer, int hidden_layer, int out_layer, double alpha) {
        this->input_layer = input_layer;
        this->hidden_layer = hidden_layer;
        this->out_layer = out_layer;
        this->alpha = alpha;
        init_weights();
    }

    void init_weights() {
        srand(time(nullptr));
        weights_0_1.resize(input_layer, vector <double> (hidden_layer));
        for(int i = 0; i < input_layer; ++i) {
            for(int j = 0; j < hidden_layer; ++j) {
                weights_0_1[i][j] = rand_double(-0.02, 0.02);
            }
        }
        weights_1_2.resize(hidden_layer, vector <double> (out_layer));
        for(int i = 0; i < hidden_layer; ++i) {
            for(int j = 0; j < out_layer; ++j) {
                weights_1_2[i][j] = rand_double(-0.02, 0.02);
            }
        }
    }

    void train(vector <vector <vector <int>>>& train_data, vector <int>& labels, int iterations, bool is_output = true, bool need_all_test = true) {
        for(int it = 1; it <= iterations; ++it) {
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
                train_data_test(train_data, labels);
            }
        }
    }

    int iteration(vector <vector <vector <int>>>& train_data, vector <int>& labels) {
        int correct_cnt = 0;
        int size = train_data.size();
        for(int cnt = 0; cnt < size; ++cnt) {
            vector <double> layer_0(input_layer);
            // Loading test data into 1-d array
            for(int i = 0; i < 28; ++i) {
                for(int j = 0; j < 28; ++j) {
                    layer_0[i * 28 + j] = train_data[cnt][i][j];
                }
            }
            vector <double> layer_1(hidden_layer, 0);
            for(int j = 0; j < hidden_layer; ++j) {
                for(int i = 0; i < input_layer; ++i) {
                    layer_1[j] += layer_0[i] * weights_0_1[i][j];
                }
            }
            for(int i = 0; i < hidden_layer; ++i) {
                layer_1[i] = relu(layer_1[i]);
            }
            vector <double> dropout_mask(hidden_layer);
            for(int i = 0; i < hidden_layer; ++i)
                dropout_mask[i] = rand() % 2;
            for(int i = 0; i < hidden_layer; ++i) {
                layer_1[i] *= dropout_mask[i] * 2;
            }
            vector <double> layer_2(out_layer, 0);
            for(int j = 0; j < out_layer; ++j) {
                for(int i = 0; i < hidden_layer; ++i) {
                    layer_2[j] += layer_1[i] * weights_1_2[i][j];
                }
            }
            int cur_ans = argmax(layer_2);
            if(cur_ans == labels[cnt])
                correct_cnt++;
            vector <double> layer_2_delta(out_layer);
            int ans_label = labels[cnt];
            for(int i = 0; i < out_layer; ++i) {
                if(i == ans_label) {
                    layer_2_delta[i] = 1 - layer_2[i];
                }
                else {
                    layer_2_delta[i] = 0 - layer_2[i];
                }
            }
            vector <double> layer_1_relu2deriv(hidden_layer);
            for(int i = 0; i < hidden_layer; ++i)
                layer_1_relu2deriv[i] = relu2deriv(layer_1[i]);
            vector <vector <double>> weights_1_2_T = transparent(weights_1_2);
            vector <double> layer_1_delta(hidden_layer, 0);
            for(int j = 0; j < hidden_layer; ++j) {
                for(int i = 0; i < out_layer; ++i) {
                    layer_1_delta[j] += layer_2_delta[i] * weights_1_2_T[i][j] * layer_1_relu2deriv[j];
                }
            }
            for(int i = 0; i < hidden_layer; ++i) {
                layer_1_delta[i] *= dropout_mask[i];
            }
            //vector <vector <double>> mul_prod_1_2(hidden_layer, vector <double> (out_layer));
            for(int i = 0; i < hidden_layer; ++i) {
                for(int j = 0; j < out_layer; ++j) {
                    weights_1_2[i][j] += layer_1[i] * layer_2_delta[j] * alpha;
                }
            }
//            for(int i = 0; i < hidden_layer; ++i) {
//                for(int j = 0; j < out_layer; ++j) {
//                    weights_1_2[i][j] += mul_prod_1_2[i][j];
//                }
//            }
//            vector <vector <double>> mul_prod_0_1(input_layer, vector <double> (hidden_layer));
            for(int i = 0; i < input_layer; ++i) {
                for(int j = 0; j < hidden_layer; ++j) {
                    weights_0_1[i][j] += layer_0[i] * layer_1_delta[j] * alpha;
                }
            }
//            for(int i = 0; i < input_layer; ++i) {
//                for(int j = 0; j < hidden_layer; ++j) {
//                    weights_0_1[i][j] += mul_prod_0_1[i][j];
//                }
//            }
        }
        return correct_cnt;
    }

    void train_data_test(vector <vector <vector <int>>>& train_data, vector <int>& labels, bool is_output = true) {
        int cnt_correct = 0;
        int size = train_data.size();
        for(int cnt = 0; cnt < size; ++cnt) {
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
                layer_1[i] = relu(layer_1[i]);
            }
            vector<double> layer_2(out_layer, 0);
            for (int j = 0; j < out_layer; ++j) {
                for (int i = 0; i < hidden_layer; ++i) {
                    layer_2[j] += layer_1[i] * weights_1_2[i][j];
                }
            }
            int cur_ans = argmax(layer_2);
            if(cur_ans == labels[cnt])
                cnt_correct++;
        }
        if(is_output) {
            cout << "       " << "Passed test in all train data: " << cnt_correct << '\n';
        }
    }

    void custom_test(vector <vector <vector <int>>>& test_data, bool is_out_sample = true, bool is_out = true) {
        int size = test_data.size();
        for(int cnt = 0; cnt < size; ++cnt) {
            vector<double> layer_0(input_layer);
            // Loading test data into 1-d array
            for (int i = 0; i < 28; ++i) {
                for (int j = 0; j < 28; ++j) {
                    layer_0[i * 28 + j] = test_data[cnt][i][j];
                }
            }
            vector<double> layer_1(hidden_layer, 0);
            for (int j = 0; j < hidden_layer; ++j) {
                for (int i = 0; i < input_layer; ++i) {
                    layer_1[j] += layer_0[i] * weights_0_1[i][j];
                }
            }
            for (int i = 0; i < hidden_layer; ++i) {
                layer_1[i] = relu(layer_1[i]);
            }
            vector<double> layer_2(out_layer, 0);
            for (int j = 0; j < out_layer; ++j) {
                for (int i = 0; i < hidden_layer; ++i) {
                    layer_2[j] += layer_1[i] * weights_1_2[i][j];
                }
            }
            if(is_out_sample) {
                output_letter(test_data[cnt]);
            }
            if(is_out) {
                int cur_ans = argmax(layer_2);
                cout << "Number in alphabet: " << cur_ans + 1 << '\n';
            }
        }
    }

private:
    double alpha;
    int input_layer, hidden_layer, out_layer;
    vector <vector <double>> weights_0_1;
    vector <vector <double>> weights_1_2;
};
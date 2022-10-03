// Portfolio 3: ML Algorithms From Scratch
// Aloksai Choudari and Alekhya Pinnamaneni
// CS 4375.003

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;
using namespace std::chrono;

// function for matrix multiplication
vector<double> matrix_mult(vector<vector<int>> m1, vector<double> m2) {
    vector<double> result(m1.size());
    for (int i = 0; i < result.size(); i++) {
        result[i] = (m1[i][0] * m2[0]) + (m1[i][1] * m2[1]);
    }
    return result;
}

// 2nd function for matrix multiplication
vector<double> matrix_mult2(vector<vector<int>> m1, vector<double> m2) {
    vector<double> result(m1.size());
    for (int i = 0; i < m2.size(); i++) {
        result[0] += m1[0][i] * m2[i];
        result[1] += m1[1][i] * m2[i];
    }
    return result;
}

// function for finding the tranpose of a matrix
vector<vector<int>> matrix_transpose(vector<vector<int>> matrix) {
    vector<vector<int>> t(matrix[0].size());
    for (int i = 0; i < t.size(); i++) {
        t[i] = vector<int>(matrix.size());
    }
    for (int i = 0; i < t.size(); i++) {
        for (int j = 0; j < t[i].size(); j++) {
            t[i][j] = matrix[j][i];
        }
    }
    return t;
}

double accuracy(double TP, double TN, double FP, double FN) {
    double acc;
    acc = (TP + TN) / (TP + TN + FP + FN);
    return acc;
}

double sensitivity(double TP, double FN) {
    double sens;
    sens = TP / (TP + FN);
    return sens;
}

double specificity(double TN, double FP) {
    double spec;
    spec = TN / (TN + FP);
    return spec;
}

int main()
{
    ifstream inFS;
    string line;
    string id_in, pclass_in, survived_in, sex_in, age_in;
    const int MAX_LEN = 1046;
    const int TRAIN_SIZE = 800;
    const int TEST_SIZE = MAX_LEN - TRAIN_SIZE;
    vector<string> id(MAX_LEN);
    vector<int> pclass(MAX_LEN);
    vector<int> survived(MAX_LEN);
    vector<int> sex(MAX_LEN);
    vector<double> age(MAX_LEN);

    cout << "Welcome to the Logistic Regression algorithm from scratch!\n" << endl;

    /*      Read in titanic data into survived and sex vectors      */
    cout << "Reading in Titanic data...\n" << endl;
    inFS.open("titanic_project.csv");   // Open file
    // Read in first line containing the headers
    getline(inFS, line);
    int numObservations = 0;
    while (numObservations <= MAX_LEN && inFS.good()) {
        // Read in each value in the row
        getline(inFS, id_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');
        // Save each value into corresponding vector
        id.at(numObservations) = id_in;
        pclass.at(numObservations) = stoi(pclass_in);
        survived.at(numObservations) = stoi(survived_in);
        sex.at(numObservations) = stoi(sex_in);
        age.at(numObservations) = stof(age_in);
        numObservations++;
    }
    inFS.close(); // Close file

    /*      Split data into train and test      */
    vector<int> train_survived(&survived[0], &survived[0] + TRAIN_SIZE);
    vector<int> test_survived(&survived[TRAIN_SIZE], &survived[TRAIN_SIZE] + TEST_SIZE);
    vector<int> train_sex(&sex[0], &sex[0] + TRAIN_SIZE);
    vector<int> test_sex(&sex[TRAIN_SIZE], &sex[TRAIN_SIZE] + TEST_SIZE);

    /*      Find logistic regression model from scratch       */
    time_point<system_clock> start = system_clock::now(); // save time before running algorithm
    // Set up weights, labels vector, and data matrix
    cout << "Calculating coefficients for logistic regression model...\n" << endl;
    vector<double> weights(2, 1);
    double diff = 1;
    double diff1 = 1;
    vector<int> labels = train_survived;
    vector<vector<int>> data_matrix(TRAIN_SIZE);
    for (int i = 0; i < TRAIN_SIZE; i++) {
        data_matrix[i] = vector<int>(2);
        // Initialize values in data matrix
        data_matrix[i][0] = 1;
        data_matrix[i][1] = train_sex[i];
    }
    // Gradient descent
    double learning_rate = 0.001;
    while (abs(diff) >= 0.00000001 || abs(diff1) >= 0.00000001) // exits loop when the change in weights becomes extremely small
    {
        vector<double> result = matrix_mult(data_matrix, weights);
        vector<double> prob_vector(TRAIN_SIZE);
        for (int i = 0; i < TRAIN_SIZE; i++) {
            prob_vector[i] = 1.0 / (1 + exp(-result[i]));
        }
        vector<double> error(TRAIN_SIZE);
        for (int i = 0; i < TRAIN_SIZE; i++) {
            error[i] = labels[i] - prob_vector[i];
        }
        vector<vector<int>> t = matrix_transpose(data_matrix);
        vector<double> mult = matrix_mult2(t, error);
        diff = weights[0];
        diff1 = weights[1];
        weights[0] += learning_rate * mult[0];
        diff -= weights[0];
        weights[1] += learning_rate * mult[1];
        diff1 -= weights[1];
    }
    time_point<system_clock> end = system_clock::now(); // save time after algorithm completes
    cout << "Coefficients:\n\tIntercept = " << (ceil(weights[0] * 10000.0) / 10000.0) << "\n\tSex = " << (ceil(weights[1] * 10000.0) / 10000.0) << "\n" << endl;

    /*      Predict survived values using test data for sex     */
    cout << "Predicting values using test data...\n" << endl;
    // Set up test labels vector and test data matrix
    vector<int> test_labels = test_survived;
    vector<vector<int>> test_matrix(TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) {
        test_matrix[i] = vector<int>(2);
        test_matrix[i][0] = 1;
        test_matrix[i][1] = test_sex[i];
    }
    // Make predictions using the weights calculated earlier
    vector<double> predicted = matrix_mult(test_matrix, weights);
    vector<double> probabilities(TEST_SIZE);
    double e = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        e = exp(predicted[i]);
        probabilities[i] = e / (1 + e);
    }
    vector<double> preds(TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) {
        if (probabilities[i] > 0.5) {
            preds[i] = 1;
        }
        else {
            preds[i] = 0;
        }
    }
    cout << "Calculating test metrics...\n" << endl;
    double TP = 0;
    double FP = 0;
    double TN = 0;
    double FN = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        if (preds[i] == 1 && test_survived[i] == 1) // true positive
        {
            TP++;
        }
        else if (preds[i] == 1 && test_survived[i] == 0) // false positive
        {
            FP++;
        }
        else if (preds[i] == 0 && test_survived[i] == 0) // true negative
        {
            TN++;
        }
        else // false negative
        {
            FN++;
        }
    }
    // Output test metrics
    cout << "Test metrics:\n\tAccuracy = " << accuracy(TP, TN, FP, FN) << "\n\tSensitivity = " << sensitivity(TP, FN) << "\n\tSpecificity = " << specificity(TN, FP) << "\n" << endl;
    // Output runtime
    duration<double> dur = end - start;
    cout << "Algorithm runtime = " << (dur).count() << "s\n" << endl;
    return 0;
}
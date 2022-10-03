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

// function to calculate the mean of age
vector<double> mean(int n, vector<double> age, vector<int> survived, vector<double> counts) {
    vector<double> mean(2, 0);
    for (int i = 0; i < n; i++) {
        if (survived[i] == 0) {
            mean[0] += age[i];
        }
        else {
            mean[1] += age[i];
        }
    }
    mean[0] /= counts[0];
    mean[1] /= counts[1];
    return mean;
}

// function to calculate the variance of age
vector<double> var(int n, vector<double> age, vector<int> survived, vector<double> counts, vector<double> mean) {
    vector<double> var(2, 0);
    for (int i = 0; i < n; i++) {
        if (survived[i] == 0) {
            var[0] += (age[i] - mean[0]) * (age[i] - mean[0]);
        }
        else {
            var[1] += (age[i] - mean[1]) * (age[i] - mean[1]);
        }
    }
    var[0] /= counts[0];
    var[1] /= counts[1];
    return var;
}

// function to calculate the likelihood of age
double lh_age(double v, double mean_v, double var_v) {
    double e = exp(-(((v - mean_v) * (v - mean_v)) / (2 * var_v)));
    return 1 / sqrt(2 * 3.141592653589793238463 * var_v) * e;
}

// function to calculate raw probabilities given pclass, sex, and age
vector<double> raw_prob(int pclass, int sex, double age, vector<vector<double>> lh_pclass, vector<vector<double>> lh_sex, vector<double> apriori, vector<double> age_mean, vector<double> age_var) {
    vector<double> raw(2, 0);
    double num_p = lh_pclass[0][pclass - 1] * lh_sex[0][sex] * apriori[0] * lh_age(age, age_mean[0], age_var[0]);
    double num_s = lh_pclass[1][pclass - 1] * lh_sex[1][sex] * apriori[1] * lh_age(age, age_mean[1], age_var[1]);
    double denom = num_p + num_s;
    raw[0] = num_p / denom;
    raw[1] = num_s / denom;
    return raw;
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

    cout << "Welcome to the Naive Bayes algorithm from scratch!\n" << endl;

    /*      Read in titanic data into survived and sex vectors      */
    cout << "Reading in Titanic data...\n" << endl;
    inFS.open("titanic_project.csv");   // Open file
    // Read in first line containing the headers
    getline(inFS, line);
    int i = 0;
    while (i <= MAX_LEN && inFS.good()) {
        // Read in each value in the row
        getline(inFS, id_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');
        // Save each value into corresponding vector
        id.at(i) = id_in;
        pclass.at(i) = stoi(pclass_in);
        survived.at(i) = stoi(survived_in);
        sex.at(i) = stoi(sex_in);
        age.at(i) = stof(age_in);
        i++;
    }
    inFS.close(); // Close file

    /*      Split data into train and test      */
    vector<int> train_pclass(&pclass[0], &pclass[0] + TRAIN_SIZE);
    vector<int> test_pclass(&pclass[TRAIN_SIZE], &pclass[TRAIN_SIZE] + TEST_SIZE);
    vector<int> train_survived(&survived[0], &survived[0] + TRAIN_SIZE);
    vector<int> test_survived(&survived[TRAIN_SIZE], &survived[TRAIN_SIZE] + TEST_SIZE);
    vector<int> train_sex(&sex[0], &sex[0] + TRAIN_SIZE);
    vector<int> test_sex(&sex[TRAIN_SIZE], &sex[TRAIN_SIZE] + TEST_SIZE);
    vector<double> train_age(&age[0], &age[0] + TRAIN_SIZE);
    vector<double> test_age(&age[TRAIN_SIZE], &age[TRAIN_SIZE] + TEST_SIZE);

    /*      Naive Bayes from scratch        */
    time_point<system_clock> start = system_clock::now(); // save time before running algorithm
    // Calculate prior probabilities
    cout << "Calculating prior probabilities...\n" << endl;
    vector<double> apriori(2, 0);
    vector<double> survivedCounts(2, 0);
    for (int i = 0; i < TRAIN_SIZE; i++) {
        if (train_survived[i] == 0) {
            survivedCounts[0] += 1;;
        }
        else {
            survivedCounts[1] += 1;
        }
    }
    apriori[0] = survivedCounts[0] / TRAIN_SIZE;
    apriori[1] = survivedCounts[1] / TRAIN_SIZE;
    cout << "Prior probabilities:\n\tPerished = " << apriori[0] << "\n\tSurvived = " << apriori[1] << "\n" << endl;

    /*      Calculate likelihood for each predictor     */
    cout << "Calculating likelihoods for pclass, sex, and age...\n" << endl;
    // Likelihood for pclass
    vector<vector<double>> lh_pclass = { { 0, 0, 0 }, { 0, 0, 0 } };
    for (int i = 0; i < TRAIN_SIZE; i++) {
        if (train_survived[i] == 0 && train_pclass[i] == 1) {
            lh_pclass[0][0] += 1;
        }
        else if (train_survived[i] == 0 && train_pclass[i] == 2) {
            lh_pclass[0][1] += 1;
        }
        else if (train_survived[i] == 0 && train_pclass[i] == 3) {
            lh_pclass[0][2] += 1;
        }
        else if (train_survived[i] == 1 && train_pclass[i] == 1) {
            lh_pclass[1][0] += 1;
        }
        else if (train_survived[i] == 1 && train_pclass[i] == 2) {
            lh_pclass[1][1] += 1;
        }
        else {
            lh_pclass[1][2] += 1;
        }
    }
    for (int i = 0; i < lh_pclass.size(); i++) {
        for (int j = 0; j < lh_pclass[i].size(); j++) {
            lh_pclass[i][j] /= survivedCounts[i];
        }
    }

    // Likelihood for sex
    vector<vector<double>> lh_sex = { { 0, 0 }, { 0, 0 } };
    for (int i = 0; i < TRAIN_SIZE; i++) {
        if (train_survived[i] == 0 && train_sex[i] == 0) {
            lh_sex[0][0] += 1;
        }
        else if (train_survived[i] == 0 && train_sex[i] == 1) {
            lh_sex[0][1] += 1;
        }
        else if (train_survived[i] == 1 && train_sex[i] == 0) {
            lh_sex[1][0] += 1;
        }
        else {
            lh_sex[1][1] += 1;
        }
    }
    for (int i = 0; i < lh_sex.size(); i++) {
        for (int j = 0; j < lh_sex[i].size(); j++) {
            lh_sex[i][j] /= survivedCounts[i];
        }
    }

    // Likelihood for age
    //Compute mean and variance for age
    vector<double> age_mean = mean(TRAIN_SIZE, train_age, train_survived, survivedCounts);
    vector<double> age_var = var(TRAIN_SIZE, train_age, train_survived, survivedCounts, age_mean);
    vector<double> age_sd(2, 0);
    age_sd[0] = sqrt(age_var[0]);
    age_sd[1] = sqrt(age_var[1]);

    time_point<system_clock> end = system_clock::now(); // save time after algorithm completes

    // Output likelihoods
    cout << "Likelihood for p(pclass|survived):\n\t\t1\t\t2\t\t3\nPerished\t" << lh_pclass[0][0] << "\t" << lh_pclass[0][1] << "\t\t" << lh_pclass[0][2] << "\nSurvived\t" << lh_pclass[1][0] << "\t" << lh_pclass[1][1] << "\t" << lh_pclass[1][2] << "\n" << endl;
    cout << "Likelihood for p(sex|survived):\n\t\tFemale\t\tMale\nPerished\t" << lh_sex[0][0] << "\t" << lh_sex[0][1] << "\nSurvived\t" << lh_sex[1][0] << "\t" << lh_sex[1][1] << "\n" << endl;
    cout << "Age:\n\t\tMean\t\tStandard Deviation\nPerished\t" << age_mean[0] << "\t\t" << age_sd[0] << "\nSurvived\t" << age_mean[1] << "\t\t" << age_sd[1] << "\n" << endl;

    /*      Make predictions using Naive Bayes model (using raw_prob() function)       */
    vector<double> probabilities(TEST_SIZE); // will store the probability that the passenger perished
    for (int i = 0; i < TEST_SIZE; i++) {
        probabilities[i] = raw_prob(test_pclass[i], test_sex[i], test_age[i], lh_pclass, lh_sex, apriori, age_mean, age_var)[0];
    }
    vector<double> preds(TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) {
        if (probabilities[i] > 0.5) {
            preds[i] = 0;
        }
        else {
            preds[i] = 1;
        }
    }
    cout << "Calculating test metrics...\n" << endl;
    double TP = 0;
    double FP = 0;
    double TN = 0;
    double FN = 0;
    for (int i = 0; i < preds.size(); i++) {
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
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>
using namespace std;

// Function to calculate sum of data
double sumFunction(vector<double> parameter) {
    double total = 0;

    // Loop to iterate through vector
    for(int i = 0; i < parameter.size(); i++) {
        total += parameter[i];
    }

    return total; // Return sum
}

// Function to calculate mean of data
double meanFunction(vector<double> parameter) {
    return (sumFunction(parameter) / parameter.size()); // Return mean
}

// Function to calculate median of data
double medianFunction(vector<double> parameter) {
    int vectorSize = parameter.size();
    
    sort(parameter.begin(), parameter.end()); // Sort the vector in ascending order

    // Check if the vector has even number of units and calculate median accordingly
    if(vectorSize % 2 == 0) {
        return (parameter[(vectorSize / 2) - 1] + parameter[vectorSize / 2]) / 2;
    }

    return parameter[vectorSize / 2]; // Return median
}

// Function to calculate range of data
double rangeFunction(vector<double> parameter) {    
    sort(parameter.begin(), parameter.end()); // Sort vector

    return parameter.back() - parameter.front(); // Return range
}

// Function to print the sum, mean, median, and range of data
void print_stats(vector<double> parameter) {
    cout << "Sum: " << sumFunction(parameter) << endl;
    cout << "Mean: " << meanFunction(parameter) << endl;
    cout << "Median: " << medianFunction(parameter) << endl;
    cout << "Range: " << rangeFunction(parameter) << endl;
}

// Function to calculate covariance of data
double covar(vector<double> parameterA, vector<double> parameterB) {
    double summation = 0;

    // Iterate through the first vector
    for(int i = 0; i < parameterA.size(); i++) {
        // Calculate the numerator of the formula
        summation += (parameterA[i] - meanFunction(parameterA)) * (parameterB[i] - meanFunction(parameterB));
    }

    return (summation / (parameterA.size() - 1)); // Return covariance (numerator divided by n - 1)
}

// Function to calculate correlation of data
double cor(vector<double> parameterA, vector<double> parameterB) {
    double variableA = sqrt(covar(parameterA, parameterA)); // Get sqauare root of parameter A calculation
    double variableB = sqrt(covar(parameterB, parameterB)); // Get sqauare root of parameter B calculation
    double covariance = covar(parameterA, parameterB); // Calculate covariance of two vectors

    return (covariance / (variableA * variableB)); // Return correlation (covariance divided by A * B after square root)
}

int main(int argc, char** argv) {
    
    ifstream inFS; // Input file stream
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    // Try to open file
    cout << "Opening file Boston.csv." << endl;

    inFS.open("Boston.csv");
    if(!inFS.is_open()) {
        cout << "Could not open file Boston.csv." << endl;
        return 1; // 1 indicates error
    }

    // Can now use inFS stream like cin stream
    // Boston.csv should contain two doubles

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    // echo heading
    cout << "heading: " << line << endl;

    int numObservations = 0;
    while (inFS.good()) {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "new length " << rm.size() << endl;

    
    cout << "Closing file Boston.csv." << endl;
    inFS.close(); // Done with file, so close it

    cout << "\nStats for rm" << endl;
    print_stats(rm);

    cout << "\nStats for medv" << endl;
    print_stats(medv);

    cout << "\nCovariance = " << covar(rm, medv) << endl;

    cout << "\nCorrelation = " << cor(rm, medv) << endl;

    cout << "\nProgram terminated.";
    
    return 0;
}
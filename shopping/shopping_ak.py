import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4



def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def find_month(month):
        switcher={
                'Jan' : 0,
                'Feb' : 1,
                'Mar' : 2,
                'Apr' : 3,
                'May' : 4,
                'June' : 5,
                'Jul' : 6,
		        'Aug' : 7,
		        'Sep' : 8,
		        'Oct' : 9,
		        'Nov' : 10,
		        'Dec' : 11 
             }
        return switcher.get(month,"Invalid month")
    

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open("shopping.csv") as f :
        reader = csv.reader(f)
        next(reader)
        
        label = []
        list_of_evidence = []

        for row in reader :
            evidence = []
            evidence.append(int(row[0]))
            evidence.append(float(row[1]))
            evidence.append(int(row[2]))
            evidence.append(float(row[3]))
            evidence.append(int(row[4]))
            evidence.append(float(row[5]))
            evidence.append(float(row[6]))
            evidence.append(float(row[7]))
            evidence.append(float(row[8]))
            evidence.append(float(row[9]))
            evidence.append(find_month(row[10]))
            evidence.append(int(row[11]))
            evidence.append(int(row[12]))
            evidence.append(int(row[13]))
            evidence.append(int(row[14]))
            if row[15] == 'Returning_Visitor' :
                evidence.append(1)
            else :
                evidence.append(0)
            if row[16] == 'TRUE' :
                evidence.append(1)
            else :
                evidence.append(0)

            list_of_evidence.append(evidence)

            if row[17] == 'TRUE' :
                label.append(1)
            else :
                label.append(0) 
        
    return (list_of_evidence,label)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    total_positive = 0
    correct_positive = 0
    incorrect_positive = 0

    total_negative = 0
    correct_negative = 0
    incorrect_negative = 0

    for actual , guess in zip(labels,predictions) :
        # Purchase was actually made
        if actual == 1 :
            total_positive += 1
            if actual == guess :
                correct_positive += 1
            else :
                incorrect_positive += 1
        else :
            total_negative += 1
            if actual == guess :
                correct_negative += 1
            else :
                incorrect_negative += 1

    sensitivity = correct_positive/total_positive
    specificity = correct_negative/total_negative

    return (sensitivity,specificity)

if __name__ == "__main__":
    main()

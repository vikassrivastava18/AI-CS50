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

    floats = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',
              'PageValues', 'SpecialDay']
    ints = ['Administrative', 'Informational', 'ProductRelated', 'Month', 'OperatingSystems', 'Browser', 'Region',
            'TrafficType', 'VisitorType', 'Weekend']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    evidence = []
    labels = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        columns = next(csvreader)

        # extracting the rows data, for evidence and label
        for row in csvreader:
            row_list = []

            for i, item in enumerate(row):

                # Floats
                if columns[i] in floats:
                    row_list.append(float(item))

                # Integers
                elif columns[i] in ints:
                    # Check if the item is a month
                    if item in months:
                        # Save the month index
                        row_list.append(int(months.index(item)))

                    elif item == 'Returning_Visitor':
                        row_list.append(int(1))

                    elif item == 'New_Visitor' or item == 'Other':
                        row_list.append(int(0))

                    elif columns[i] == 'Weekend':
                        if item == 'TRUE':
                            row_list.append(int(1))
                        elif item == 'FALSE':
                            row_list.append(int(0))
                    else:
                        row_list.append(int(item))

            evidence.append(row_list)

            # Add the label for the row
            if row[-1] == 'FALSE':
                labels.append(0)
            if row[-1] == 'TRUE':
                labels.append(1)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=5)
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

    true_positive,  positive= 0, 0
    true_negative, negative = 0, 0
    for i, label in enumerate(labels):
        if label == 0:
            if predictions[i] == label:
                true_negative += 1
            negative += 1
        elif label == 1:
            if predictions[i] == label:
                true_positive += 1
            positive += 1

    true_positive_rate = true_positive / positive
    true_negative_rate = true_negative / negative

    return (true_positive_rate, true_negative_rate)

if __name__ == "__main__":
    main()

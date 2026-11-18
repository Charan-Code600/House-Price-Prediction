




from src.config import MODEL_PATH, NEIGHBORHOODS
from src.data_loader import clean_na_values, data_quality_report, load_raw_data
from src.eda import (
    plot_area_vs_price,
    plot_neighborhood_prices,
    plot_saleprice_distribution,
    plot_top_correlations,
)
from src.evaluate import (
    evaluate_on_test,
    plot_actual_vs_predicted,
    plot_feature_importance,
    worst_errors,
)
from src.predict import load_model, predict_price
from src.preprocess import split_data
from src.train import compare_models, save_model, tune_models

MENU = """
1. Data Quality Report
2. Create EDA Graphs
3. Compare Models (Cross-Validation)
4. Train, Evaluate and Save Final Model
5. Predict a House Price
6. Exit
"""


def ask_number(text, low, high):
    while True:
        value = input(text)
        if value.isdigit() and low <= int(value) <= high:
            return int(value)
        print(f"Please enter a whole number between {low} and {high}.")


def ask_neighborhood():
    lookup = {name.lower(): name for name in NEIGHBORHOODS}
    while True:
        value = input("Neighborhood (type ? to see the list): ").strip()
        if value == "?":
            print(", ".join(NEIGHBORHOODS))
        elif value.lower() in lookup:
            return lookup[value.lower()]
        else:
            print("Unknown neighborhood. Type ? to see valid names.")


def run_data_quality():
    report = data_quality_report(clean_na_values(load_raw_data()))
    print("Rows           :", report["rows"])
    print("Columns        :", report["columns"])
    print("Duplicate rows :", report["duplicate_rows"])
    print("Columns with real missing values:")
    print(report["missing"].to_string())


def run_eda():
    df = clean_na_values(load_raw_data())
    plot_saleprice_distribution(df)
    plot_area_vs_price(df)
    plot_top_correlations(df)
    plot_neighborhood_prices(df)
    print("EDA graphs saved in reports/figures")


def run_compare():
    df = clean_na_values(load_raw_data())
    X_train, X_test, y_train, y_test = split_data(df)
    print("Running 5-fold cross-validation, please wait...")
    print(compare_models(X_train, y_train).to_string())


def run_train_and_save():
    df = clean_na_values(load_raw_data())
    X_train, X_test, y_train, y_test = split_data(df)
    print("Tuning models, please wait...")
    results, best_models = tune_models(X_train, y_train)
    print(results.to_string())

    final_model = best_models["Lasso"]
    print()
    print("TEST RESULTS (final model: Lasso)")
    for name, value in evaluate_on_test(final_model, X_test, y_test).items():
        print(f"  {name}: {value}")
    print()
    print("10 biggest mistakes on test data:")
    print(worst_errors(final_model, X_test, y_test).to_string())

    plot_actual_vs_predicted(final_model, X_test, y_test)
    plot_feature_importance(final_model)
    save_model(final_model)
    print()
    print("Graphs saved in reports/figures. Model saved in models/.")


def run_predict():
    if not MODEL_PATH.exists():
        print("No saved model found. Please run option 4 first.")
        return
    model = load_model()
    house = {
        "GrLivArea": ask_number("Living area in sq ft (300-6000): ", 300, 6000),
        "OverallQual": ask_number("Overall quality (1-10): ", 1, 10),
        "YearBuilt": ask_number("Year built (1870-2010): ", 1870, 2010),
        "Neighborhood": ask_neighborhood(),
    }
    price = predict_price(model, house)
    print(f"Estimated price: ${price:,.0f}")
    print("(Estimate uses only these 4 details; other details are assumed typical.)")


def main():
    print(MENU)
    while True:
        print("*" * 50)
        option = input("choose: ")

        if option == "1":
            run_data_quality()
        elif option == "2":
            run_eda()
        elif option == "3":
            run_compare()
        elif option == "4":
            run_train_and_save()
        elif option == "5":
            run_predict()
        elif option == "6":
            print("Bye!")
            break
        else:
            print("Invalid option!")


if __name__ == "__main__":
    main()



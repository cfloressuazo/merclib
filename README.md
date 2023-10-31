
# Mercadolibre Item Classifier Project Readme
This project is a small program designed to process a dataset and train a
binary classifier for classifying Mercadolibre items posted in the marketplace
as either "new" or "used". Additionally, it provides a testing mechanism to
evaluate the classifier's performance with metrics, targeting an accuracy of
approximately 0.87 and precision of 0.9.

## Installation
To get started, follow these installation steps:

1. Ensure you have Poetry installed on your system.
2. Clone the project repository from the source:
```bash
git clone <repository_url>
```
3. Navigate to the project's root directory:
```bash
cd project_directory
```
4. Install project dependencies using Poetry:
```bash
poetry install
```

## Running the Project
You can run the project using the following command:

```bash
poetry shell
poetry run python3 src/new_used/main.py -f path/to/dataset.json
```
If you run the script without the `-f` flag, it will use the default dataset
located at `root_project_dir/data/MLA_100k_checked_v3.jsonlines`.

## Runnin the Project with Docker
You can run the project via Docker using the following command:

```bash
docker build . -t merclib:latest
docker run
```

## Project Structure
The project is structured as follows:
* `src/`: Contains the source code for the Mercadolibre item classifier.
* `data/`: This directory holds the dataset used for training and testing.
* `src/new_used/main.py`: The main script for training and testing the classifier.
* `notebooks`: There are two notebooks one for the EDA and another for choosing the
* right model algorithm, hyperparamters and features.

## Dataset
Ensure that your dataset is in JSON format, with the necessary attributes for classification.

## Results and Metrics
After running the project, you will receive classification results, including accuracy and
precision (and other performance metrics). The goal is to achieve an accuracy of around 0.87
and a precision of 0.9. Make sure to inspect these metrics to assess the performance of the
classifier.

* The primary metric chosen is **Acuraccy** with a *.87*
* The secondary metric chosen is **Precision** with *0.9*. The reason to choose this metric
  is because if be ause to me it is more relevant to avoid false positives when classiying
  items in the marketplace. Users buying an item as "new" from the platform and then getting
  a "used" items afterwards is likely to be considered as a very bad experience so to me as
  a client is better to have mercadolibre to label correctly new items (so focusing on minimizing
  False Positives)
* Results are displayed below:
```
** Accuracy score of classifier:  0.8694
** Precision score of classifier:  0.8969790859798605
** Recall score of classifier:  0.8568257491675916
** F1 score of classifier:  0.8764427625354777
```

Please refer to the project documentation for additional details and customization options.

## Contributors
Cesar Flores - cfloressuazo@gmail.com

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

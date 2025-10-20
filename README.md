# Text Summariser / Complaint Category Classifier

This repository contains a Jupyter notebook, `text-summariser.ipynb`, that demonstrates a full pipeline for classifying consumer complaint narratives into broad categories (Credit reporting, Debt collection, Consumer Loan, Mortgage). The notebook includes data loading, preprocessing, TF-IDF vectorization, model training (Logistic Regression, MultinomialNB, Linear SVC), evaluation, and a small prediction demo.

## Files
- `text-summariser.ipynb` - The main notebook with the full pipeline and explanatory cells.
- `complaints.csv` - dataset file expected in the repository root or external path; the notebook originally referenced a Kaggle path in the example.

## Description
The notebook performs the following steps:
1. Installs and imports necessary libraries (NLTK, scikit-learn, pandas, seaborn, matplotlib).
2. Downloads required NLTK assets (stopwords, wordnet).
3. Loads the complaints dataset and filters products into mapped category IDs.
4. Cleans and lemmatizes complaint text, removes stopwords and non-letter characters.
5. Vectorizes text using TF-IDF and trains multiple classifiers.
6. Evaluates models using accuracy, classification reports and confusion matrix.
7. Provides a helper function to predict the category of new complaint text.

## Dependencies
The notebook uses the following Python libraries:
- python (3.8+ recommended)
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk

You can install them with pip. Example:

```powershell
python -m pip install --upgrade pip; 
python -m pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

After installing, the notebook also downloads NLTK corpora at runtime (stopwords and wordnet):

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

(Note: there is a small typo above in the notebook's example: `nltk.download('wordnet')` should be used — the README has the correct spelling in the dependencies section.)

## How to run
1. Open `text-summariser.ipynb` in Jupyter Notebook / JupyterLab / VS Code's Notebook interface.
2. Ensure the dataset path in the notebook points to the actual `complaints.csv` location on your machine.
   - The notebook currently reads: `pd.read_csv('/kaggle/input/complaints-data/complaints.csv')`.
   - If `complaints.csv` is in the repository root, change it to: `pd.read_csv('complaints.csv')`.
3. Run each cell in order. The notebook will install NLTK (if not present) and download required corpora.
4. Inspect model outputs and try the prediction helper at the end of the notebook.

## Dataset notes
- The sample notebook filters `Product` values to a custom `category_map`. If your dataset has different product names, update the `category_map` dictionary accordingly.
- The notebook drops rows with missing narratives. Review `df.dropna(...)` if you need different behavior.

## Troubleshooting
- If plotting cells fail in VS Code, ensure matplotlib is configured for inline plotting or use `%matplotlib inline` at the top of the notebook.
- If you get Unicode/encoding errors while reading `complaints.csv`, try specifying `encoding='utf-8'` or `encoding='latin-1'` in `pd.read_csv`.
- If NLTK downloads fail due to network/firewall, download the corpora manually using NLTK's downloader GUI, or place the corpora in the NLTK data path.

## Next steps (suggested)
- Extract dependencies to a `requirements.txt` for reproducible installs.
- Add a small example script (e.g., `predict.py`) to use the trained model outside the notebook.
- Save the best model with joblib/pickle and demonstrate loading it for predictions.

## License
Add a license if you plan to share the project publicly.

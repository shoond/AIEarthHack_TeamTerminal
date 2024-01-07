# pip install nltk
import nltk
nltk.download('all')
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('https://raw.githubusercontent.com/shoond/portfolio/datasets/AI%20EarthHack%20Dataset.csv', encoding='Latin-1').dropna(subset=['problem', 'solution'])
df.solution = df.solution.astype(str)

# create preprocess_text function
def preprocess_text(text):

    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

# apply the function df

df['problem_cleaned'] = df['problem'].apply(preprocess_text)
df['solution_cleaned'] = df['solution'].apply(preprocess_text)

sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis on the 'solution' column
df['solution_sentiment'] = df['solution_cleaned'].apply(lambda x: sia.polarity_scores(x)['compound'])

#Slice to include sentiment scores > 0.90. We imply NLTK sentiment analysis >0.90 is sufficient in exploratory analysis of problem-solution pairs.
df_90 = df[df['solution_sentiment'] > 0.90]
df_90

# Save DataFrame to a CSV file
df_90.to_csv('AI EarthHack Dataset_SA90.csv', index=False)

import tkinter as tk
from tkinter import Text, Scrollbar
import pandas as pd

class CSVViewer:
    def __init__(self, root, data):
        self.root = root
        self.data = data
        self.current_index = 0
        self.top_10_rows = []

        self.root.geometry("800x600")  # Set window size

        self.text_widget = Text(root, wrap="word", font=("Arial", 12), width=80, height=20)
        self.text_widget.pack(side=tk.LEFT, fill=tk.Y)

        self.scrollbar = Scrollbar(root, command=self.text_widget.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.config(yscrollcommand=self.scrollbar.set)

        self.display_top_10()

        self.prev_button = tk.Button(root, text="Previous", command=self.show_previous)
        self.prev_button.pack()
        self.next_button = tk.Button(root, text="Next", command=self.show_next)
        self.next_button.pack()

    def display_top_10(self):
        self.top_10_rows = self.data.sort_values(by='solution_sentiment', ascending=False).head(10).to_dict('records')
        self.display_row()

    def display_row(self):
        if 0 <= self.current_index < len(self.top_10_rows):
            current_row = self.top_10_rows[self.current_index]
            display_text = f"\nProblem: {current_row['problem']}\n\nSolution: {current_row['solution']}\n\nIdea Rating: {current_row['solution_sentiment']}\n\n"

            self.text_widget.delete('1.0', tk.END)  # Clear previous text
            self.text_widget.insert(tk.END, display_text)

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_row()

    def show_next(self):
        if self.current_index < len(self.top_10_rows) - 1:
            self.current_index += 1
            self.display_row()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("AI EarthHack Evaluator GUI")

    data = pd.read_csv('AI EarthHack Dataset_SA90.csv', encoding='Latin-1')

    csv_viewer = CSVViewer(root, data)

    root.mainloop()

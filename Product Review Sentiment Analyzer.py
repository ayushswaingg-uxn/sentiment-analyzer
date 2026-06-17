import argparse
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

sns.set_style("whitegrid")
PLOTS_DIR = "sentiment_plots"
latest_df = None

# Sample Dataset
SAMPLE_REVIEWS = [
    ("Amazing build quality, the phone feels premium and the camera is outstanding!", "Electronics"),
    ("Battery drains very fast, extremely disappointed with this purchase.", "Electronics"),
    ("It's okay, does the job but nothing special about it.", "Electronics"),
    ("Best purchase I have made this year, super fast delivery too!", "Electronics"),
    ("Worst product ever, stopped working within two days.", "Electronics"),
    ("The fabric is soft and the stitching is neat, loved the fit.", "Fashion"),
    ("Color faded after the first wash, not worth the price.", "Fashion"),
    ("Average quality, similar to what you get in local stores.", "Fashion"),
    ("Super comfortable shoes, wearing them every day now.", "Fashion"),
    ("Size chart is misleading, had to return the item.", "Fashion"),
    ("Great sound quality and the noise cancellation works perfectly.", "Electronics"),
    ("Packaging was damaged and one earbud was not working.", "Electronics"),
    ("Decent product for the price range, no complaints.", "Electronics"),
    ("Exceeded my expectations, will definitely buy again.", "Home"),
    ("Customer service was rude and unhelpful when I raised a complaint.", "Home"),
    ("It works fine, nothing extraordinary but it gets the job done.", "Home"),
    ("Absolutely love it, the design is elegant and functional.", "Home"),
    ("Received a used item instead of a new one, very disappointed.", "Home"),
    ("The taste is good but the packaging could be better.", "Grocery"),
    ("Fresh and delivered on time, satisfied with the purchase.", "Grocery"),
    ("Item was expired when it arrived, wasted my money.", "Grocery"),
    ("Not bad, could be improved but overall a fair deal.", "Grocery"),
    ("Five stars! Exactly as described and arrived earlier than expected.", "Books"),
    ("The pages were torn and the cover was scratched, poor packaging.", "Books"),
    ("A good read, kept me engaged till the end.", "Books"),
    ("Overpriced for the content, expected much more.", "Books"),
    ("Setup was simple and instructions were clear, happy with it.", "Home"),
    ("This is the second time I am facing the same issue, very frustrating.", "Electronics"),
    ("Quality matches the price, nothing more nothing less.", "Fashion"),
    ("Highly recommend this product, value for money!", "Grocery"),
]

PALETTE = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral": "#95a5a6"
}

# ----------------------------
# Data Loading
# ----------------------------
def load_sample_data():
    return pd.DataFrame(SAMPLE_REVIEWS, columns=["review", "category"])


def load_csv_data(path, column):
    df = pd.read_csv(path)

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found.\nAvailable columns: {list(df.columns)}"
        )

    df = df.rename(columns={column: "review"})
    df = df.dropna(subset=["review"]).reset_index(drop=True)
    return df


# ----------------------------
# Text Cleaning
# ----------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# ----------------------------
# Sentiment Models
# ----------------------------
vader_analyzer = SentimentIntensityAnalyzer()


def vader_sentiment(text):
    score = vader_analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, score


def textblob_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, polarity


# ----------------------------
# Analysis
# ----------------------------
def analyze_dataframe(df):
    df = df.copy()

    df["clean_review"] = df["review"].apply(clean_text)

    vader_results = df["review"].apply(vader_sentiment)
    df["vader_sentiment"] = vader_results.apply(lambda x: x[0])
    df["vader_score"] = vader_results.apply(lambda x: x[1])

    tb_results = df["review"].apply(textblob_sentiment)
    df["textblob_sentiment"] = tb_results.apply(lambda x: x[0])
    df["textblob_score"] = tb_results.apply(lambda x: x[1])

    df["agreement"] = (
        df["vader_sentiment"] == df["textblob_sentiment"]
    )

    return df


# ----------------------------
# Summary
# ----------------------------
def get_summary_text(df):
    total = len(df)
    lines = []

    lines.append("=" * 60)
    lines.append("SENTIMENT ANALYSIS SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total Reviews: {total}")
    lines.append("")

    for engine, col in [
        ("VADER", "vader_sentiment"),
        ("TextBlob", "textblob_sentiment"),
    ]:
        lines.append(f"{engine} Results")

        counts = df[col].value_counts()

        for label in ["Positive", "Negative", "Neutral"]:
            count = counts.get(label, 0)
            pct = (count / total) * 100 if total else 0
            lines.append(
                f"{label:<10}: {count:>3} ({pct:.1f}%)"
            )

        lines.append("")

    agreement = df["agreement"].mean() * 100 if total else 0
    lines.append(f"Agreement: {agreement:.1f}%")
    lines.append("=" * 60)

    return "\n".join(lines)


def print_summary(df):
    print(get_summary_text(df))


def format_review_details(df):
    lines = []

    for idx, row in df.reset_index(drop=True).iterrows():
        lines.append(f"Review {idx + 1}")
        lines.append(f"Category: {row.get('category', 'N/A')}")
        lines.append(f"Text: {row['review']}")
        lines.append(
            f"VADER: {row['vader_sentiment']} ({row['vader_score']:.3f})"
        )
        lines.append(
            f"TextBlob: {row['textblob_sentiment']} ({row['textblob_score']:.3f})"
        )
        lines.append(f"Agreement: {'Yes' if row['agreement'] else 'No'}")
        lines.append("-" * 80)

    return "\n".join(lines)


def show_review_details():
    global latest_df

    if latest_df is None:
        messagebox.showwarning(
            "No Analysis",
            "Run sentiment analysis first to view review details."
        )
        return

    details = format_review_details(latest_df)
    detail_window = tk.Toplevel()
    detail_window.title("Review Details")
    detail_window.geometry("900x600")

    text_box = scrolledtext.ScrolledText(
        detail_window,
        width=110,
        height=35,
        wrap=tk.WORD,
    )
    text_box.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
    text_box.insert(tk.END, details)
    text_box.config(state=tk.DISABLED)


def browse_csv_file(entry):
    path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if path:
        entry.delete(0, tk.END)
        entry.insert(0, path)


def run_gui_analysis(input_entry, column_entry, output_entry, summary_box):
    global latest_df

    data_path = input_entry.get().strip()
    review_column = column_entry.get().strip() or "review"
    output_path = output_entry.get().strip() or "analyzed_reviews.csv"

    try:
        if data_path:
            df = load_csv_data(data_path, review_column)
        else:
            df = load_sample_data()
    except Exception as exc:
        messagebox.showerror("Data Load Error", str(exc))
        return

    df = analyze_dataframe(df)
    latest_df = df
    summary_text = get_summary_text(df)

    try:
        generate_all_visualizations(df)
    except Exception as exc:
        messagebox.showerror("Visualization Error", str(exc))
        return

    try:
        df.to_csv(output_path, index=False)
    except Exception as exc:
        messagebox.showerror("File Save Error", str(exc))
        return

    full_message = (
        summary_text
        + "\n\nCharts saved in '"
        + PLOTS_DIR
        + "' folder.\nResults saved to: "
        + output_path
    )

    summary_box.config(state=tk.NORMAL)
    summary_box.delete("1.0", tk.END)
    summary_box.insert(tk.END, full_message)
    summary_box.config(state=tk.DISABLED)

    messagebox.showinfo("Analysis Complete", "Sentiment analysis complete.")


def create_gui():
    root = tk.Tk()
    root.title("Product Review Sentiment Analyzer")
    root.geometry("760x640")

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(frame, text="CSV File Path:", anchor="w").grid(row=0, column=0, sticky="w")
    input_entry = tk.Entry(frame, width=70)
    input_entry.grid(row=0, column=1, padx=(6, 0), pady=4, sticky="w")
    tk.Button(
        frame,
        text="Browse...",
        command=lambda: browse_csv_file(input_entry)
    ).grid(row=0, column=2, padx=(6, 0), pady=4)

    tk.Label(frame, text="Review Column Name:", anchor="w").grid(row=1, column=0, sticky="w")
    column_entry = tk.Entry(frame, width=30)
    column_entry.insert(0, "review")
    column_entry.grid(row=1, column=1, padx=(6, 0), pady=4, sticky="w")

    tk.Label(frame, text="Output CSV File:", anchor="w").grid(row=2, column=0, sticky="w")
    output_entry = tk.Entry(frame, width=30)
    output_entry.insert(0, "analyzed_reviews.csv")
    output_entry.grid(row=2, column=1, padx=(6, 0), pady=4, sticky="w")

    analyze_button = tk.Button(
        frame,
        text="Analyze Reviews",
        width=20,
        command=lambda: run_gui_analysis(
            input_entry,
            column_entry,
            output_entry,
            summary_box,
        ),
    )
    analyze_button.grid(row=3, column=1, pady=(10, 12), sticky="w")

    details_button = tk.Button(
        frame,
        text="Show Review Details",
        width=20,
        command=show_review_details,
    )
    details_button.grid(row=3, column=0, pady=(10, 12), sticky="w")

    sample_button = tk.Button(
        frame,
        text="Use Sample Data",
        width=20,
        command=lambda: run_gui_analysis(
            input_entry,
            column_entry,
            output_entry,
            summary_box,
        ),
    )
    sample_button.grid(row=3, column=2, pady=(10, 12), sticky="w")

    tk.Label(frame, text="Summary Output:", anchor="w").grid(row=4, column=0, columnspan=3, sticky="w")
    summary_box = scrolledtext.ScrolledText(
        frame,
        width=88,
        height=28,
        wrap=tk.WORD,
        state=tk.DISABLED,
    )
    summary_box.grid(row=5, column=0, columnspan=3, pady=(4, 0), sticky="nsew")

    frame.grid_rowconfigure(5, weight=1)
    frame.grid_columnconfigure(1, weight=1)

    tk.Label(
        frame,
        text="Leave CSV path blank to analyze built-in sample reviews.",
        fg="#555555",
        anchor="w",
    ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(10, 0))

    root.mainloop()


# ----------------------------
# Visualizations
# ----------------------------
def plot_sentiment_bar(df, column, title, filename):
    order = ["Positive", "Neutral", "Negative"]

    counts = (
        df[column]
        .value_counts()
        .reindex(order)
        .fillna(0)
    )

    plt.figure(figsize=(6, 4))

    bars = plt.bar(
        counts.index,
        counts.values,
        color=[PALETTE[c] for c in counts.index]
    )

    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            value + 0.2,
            int(value),
            ha="center"
        )

    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOTS_DIR, filename),
        dpi=150
    )

    plt.close()


def plot_sentiment_pie(df, column, title, filename):
    order = ["Positive", "Neutral", "Negative"]

    counts = (
        df[column]
        .value_counts()
        .reindex(order)
        .fillna(0)
    )

    counts = counts[counts > 0]

    plt.figure(figsize=(5, 5))

    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[PALETTE[c] for c in counts.index],
        startangle=90
    )

    plt.title(title)

    plt.savefig(
        os.path.join(PLOTS_DIR, filename),
        dpi=150
    )

    plt.close()


def plot_engine_comparison(df, filename):
    order = ["Positive", "Neutral", "Negative"]

    vader_counts = (
        df["vader_sentiment"]
        .value_counts()
        .reindex(order)
        .fillna(0)
    )

    tb_counts = (
        df["textblob_sentiment"]
        .value_counts()
        .reindex(order)
        .fillna(0)
    )

    x = range(len(order))
    width = 0.35

    plt.figure(figsize=(7, 4))

    plt.bar(
        [i - width/2 for i in x],
        vader_counts.values,
        width,
        label="VADER"
    )

    plt.bar(
        [i + width/2 for i in x],
        tb_counts.values,
        width,
        label="TextBlob"
    )

    plt.xticks(list(x), order)

    plt.title("VADER vs TextBlob")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOTS_DIR, filename),
        dpi=150
    )

    plt.close()


def plot_wordclouds(df, filename):
    if not WORDCLOUD_AVAILABLE:
        print("WordCloud package not installed.")
        return

    positive_text = " ".join(
        df[df["vader_sentiment"] == "Positive"]["clean_review"]
    )

    negative_text = " ".join(
        df[df["vader_sentiment"] == "Negative"]["clean_review"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, text, title, cmap in [
        (axes[0], positive_text, "Positive", "Greens"),
        (axes[1], negative_text, "Negative", "Reds")
    ]:

        if text.strip():
            wc = WordCloud(
                width=600,
                height=400,
                background_color="white",
                colormap=cmap
            ).generate(text)

            ax.imshow(wc)

        ax.axis("off")
        ax.set_title(title)

    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOTS_DIR, filename),
        dpi=150
    )

    plt.close()


def generate_all_visualizations(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_sentiment_bar(
        df,
        "vader_sentiment",
        "VADER Sentiment Distribution",
        "vader_bar.png"
    )

    plot_sentiment_pie(
        df,
        "vader_sentiment",
        "VADER Sentiment Share",
        "vader_pie.png"
    )

    plot_engine_comparison(
        df,
        "comparison.png"
    )

    plot_wordclouds(
        df,
        "wordclouds.png"
    )

    print(
        f"\nCharts saved in '{PLOTS_DIR}' folder."
    )


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="CSV file path"
    )

    parser.add_argument(
        "--column",
        type=str,
        default="review",
        help="Review column name"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="analyzed_reviews.csv"
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Tkinter GUI"
    )

    args = parser.parse_args()

    if args.gui:
        create_gui()
        return

    if args.input:
        df = load_csv_data(
            args.input,
            args.column
        )
    else:
        df = load_sample_data()

    print(
        f"Loaded {len(df)} reviews."
    )

    df = analyze_dataframe(df)

    print_summary(df)

    generate_all_visualizations(df)

    df.to_csv(
        args.output,
        index=False
    )

    print(
        f"\nResults saved to: {args.output}"
    )


if __name__ == "__main__":
    main()

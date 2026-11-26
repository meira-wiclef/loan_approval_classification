from collections import Counter  # counts how often each class label appears
import ctypes  # lets Windows render tkinter crisply on high-DPI displays
import sys  # checks operating system for DPI setup
import tkinter as tk  # builds the desktop graphical interface
from tkinter import messagebox  # provides pop-up alerts for invalid entries
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # embeds matplotlib figures inside tkinter
from matplotlib.figure import Figure  # creates figure objects without relying on global pyplot state
import numpy as np  # used for vector squaring
import pandas as pd  # used to extract csv
from sklearn.metrics import accuracy_score, confusion_matrix  # evaluates accuracy and right/wrong counts
from sklearn.preprocessing import StandardScaler  # standardizes features so larger scales don't dominate distance


def enable_high_dpi():
    # makes tkinter crisp instead of stretched/grainy on scaled Windows screens
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def calculate_euclidean_distance(first_point, second_point):
    first_array = np.array(first_point)
    second_array = np.array(second_point)
    squared_differences = (first_array - second_array) ** 2
    return np.sqrt(np.sum(squared_differences))


def get_distance_from_record(distance_record):
    return distance_record[0]


def predict_knn_label(train_features, train_targets, target_point, num_neighbors):
    neighbor_distances = []

    for index in range(len(train_features)):
        distance = calculate_euclidean_distance(target_point, train_features[index])
        actual_label = train_targets[index]
        neighbor_distances.append((distance, actual_label))

    neighbor_distances.sort(key=get_distance_from_record)

    nearest_records = neighbor_distances[:num_neighbors]
    nearest_labels = []
    for distance, label in nearest_records:
        nearest_labels.append(label)

    label_frequency = Counter(nearest_labels)
    most_common_label = label_frequency.most_common(1)[0][0]
    return most_common_label


def get_neighbor_vote_counts(train_features, train_targets, target_point, num_neighbors):
    neighbor_distances = []
    for index in range(len(train_features)):
        distance = calculate_euclidean_distance(target_point, train_features[index])
        neighbor_distances.append((distance, train_targets[index]))
    neighbor_distances.sort(key=get_distance_from_record)
    top_k_labels = [label for _, label in neighbor_distances[:num_neighbors]]
    counts = Counter(top_k_labels)
    return counts.get(1, 0), counts.get(0, 0)


def draw_rounded_rect(canvas, x1, y1, x2, y2, radius=12, **kwargs):
    r = max(2, min(radius, abs(x2 - x1) / 2, abs(y2 - y1) / 2))
    points = [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=12, **kwargs)


status_names = {0: "Rejected (0)",
                1: "Approved (1)"
}

# 1. loading and preparing the dataset
csv_filename = "loan_data.csv"
raw_dataset = pd.read_csv(csv_filename)  # this is a pandas dataframe

income_column = "person_income"  # feature 1
loan_amount_column = "loan_amnt"  # feature 2
credit_score_column = "credit_score"  # feature 3
target_column = "loan_status"  # the column we are trying to predict
feature_columns = [income_column, loan_amount_column, credit_score_column]  # this is a list of strings

cleaned_dataset = raw_dataset.copy()
for col in feature_columns + [target_column]:  # list concatenation, so we can clean all 4 columns at once
    # converts each column to numbers and converts any invalid ones to NaN
    cleaned_dataset[col] = pd.to_numeric(cleaned_dataset[col], errors="coerce")

# dropping invalid rows and keeping a small sample of data
loan_dataset = cleaned_dataset.dropna(subset=feature_columns + [target_column]).head(600).sample(frac=1, random_state=42)
total_records = len(loan_dataset)

print("Usable loan records:", total_records)

# splitting into training data and test data
training_count = int(0.8 * total_records)
training_records = loan_dataset.iloc[:training_count]
testing_records = loan_dataset.iloc[training_count:]

training_percentage = (len(training_records) / total_records) * 100
testing_percentage = (len(testing_records) / total_records) * 100

print("Training set:", len(training_records), "records (" + str(round(training_percentage)) + "%)",)
print("Testing set:", len(testing_records), "records (" + str(round(testing_percentage)) + "%)",)

raw_train_features = training_records[feature_columns].values.tolist()  # 2D array of applicants
train_targets = training_records[target_column].astype(int).tolist()  # 1D array of loan statuses

raw_test_features = testing_records[feature_columns].values.tolist()
test_targets = testing_records[target_column].astype(int).tolist()

# fitting StandardScaler on training features only to avoid data leakage
scaler = StandardScaler()
train_features = scaler.fit_transform(raw_train_features).tolist()
test_features = scaler.transform(raw_test_features).tolist()

# 2. using elbow method to test for best k
k_values = list(range(1, 9))
accuracies = []

for k in k_values:
    predicted_targets = [
        predict_knn_label(train_features, train_targets, test_point, k)
        for test_point in test_features
    ]
    accuracy = accuracy_score(test_targets, predicted_targets)
    accuracies.append(accuracy)

best_k = k_values[int(np.argmax(accuracies))]
print("Best k from elbow evaluation:", best_k)

# 3. evaluating accuracy with scikit learn and summarizing outcomes
best_predicted_targets = [
    predict_knn_label(train_features, train_targets, current_test_point, best_k)
    for current_test_point in test_features
]
model_accuracy = accuracy_score(test_targets, best_predicted_targets)
print(f"Model Accuracy (k = {best_k}):", round(100 * model_accuracy, 3))

cm = confusion_matrix(test_targets, best_predicted_targets, labels=[0, 1])
right_count = int(cm[0, 0] + cm[1, 1])
wrong_count = int(cm[0, 1] + cm[1, 0])

avg_income = float(training_records[income_column].mean())
avg_loan = float(training_records[loan_amount_column].mean())
avg_credit = float(training_records[credit_score_column].mean())
typical_lti_pct = float(((training_records[loan_amount_column] / training_records[income_column].replace(0, np.nan)).dropna() * 100.0).mean())

# sleek modern dark palette: midnight slate, soft lavender-violet, mint-emerald, and warm amber
BG_SLATE = "#0c0f17"
CARD_SLATE = "#141925"
CARD_ELEVATED = "#1c2333"
CARD_HOVER = "#242d42"
INPUT_SLATE = "#0f131d"
BORDER_SOFT = "#2b354d"
MINT_ACCENT = "#34d399"
MINT_DEEP = "#059669"
MINT_TINT = "#0d2b24"
AMBER_ACCENT = "#fbbf24"
AMBER_TINT = "#2e220b"
VIOLET_ACCENT = "#a78bfa"
TEXT_BRIGHT = "#f8fafc"
TEXT_MUTED = "#94a3b8"
GRID_SLATE = "#232c40"

enable_high_dpi()

root = tk.Tk()
root.title("ClarityPay // Applicant Loan & Simulation Studio")
root.geometry("1340x860")
root.minsize(1160, 780)
root.configure(bg=BG_SLATE)

outer = tk.Frame(root, bg=BG_SLATE, padx=16, pady=14)
outer.pack(fill=tk.BOTH, expand=True)

# top header + interactive hover KPI pills
header_card = tk.Frame(
    outer,
    bg=CARD_SLATE,
    padx=16,
    pady=12,
    highlightbackground=BORDER_SOFT,
    highlightthickness=1,
)
header_card.pack(fill=tk.X, pady=(0, 12))

top_row = tk.Frame(header_card, bg=CARD_SLATE)
top_row.pack(fill=tk.X)

tk.Label(
    top_row,
    text="CLARITYPAY  //  APPLICANT ELIGIBILITY & SIMULATION",
    font=("Segoe UI", 13, "bold"),
    fg=TEXT_BRIGHT,
    bg=CARD_SLATE,
).pack(side=tk.LEFT)

tk.Label(
    top_row,
    text=f"k = {best_k}  •  {total_records} sample records  •  {round(100 * model_accuracy, 1)}% accuracy",
    font=("Segoe UI", 9, "bold"),
    fg=VIOLET_ACCENT,
    bg=CARD_ELEVATED,
    padx=12,
    pady=4,
).pack(side=tk.RIGHT)

hover_bar = tk.Label(
    header_card,
    text="Hover over KPI pills or chart bars for live details — enter your numbers or run a what-if check.",
    font=("Segoe UI", 9),
    fg=TEXT_MUTED,
    bg=CARD_SLATE,
)
hover_bar.pack(anchor="w", pady=(3, 8))

kpi_row = tk.Frame(header_card, bg=CARD_SLATE)
kpi_row.pack(fill=tk.X)

kpi_items = [
    ("TEST ACCURACY", f"{round(100 * model_accuracy, 1)}%", MINT_ACCENT, f"{right_count} right and {wrong_count} wrong out of {right_count + wrong_count} test applicants."),
    ("TYPICAL INCOME", f"${avg_income:,.0f}", VIOLET_ACCENT, "Average annual income among typical training applicants."),
    ("TYPICAL LOAN", f"${avg_loan:,.0f}", AMBER_ACCENT, "Average requested loan amount among typical training applicants."),
    ("TYPICAL CREDIT", f"{avg_credit:,.0f}", MINT_ACCENT, "Average credit score among typical training applicants."),
    ("TYPICAL LTI", f"{typical_lti_pct:.1f}%", TEXT_BRIGHT, "Typical loan-to-income ratio across training data."),
]

for col_idx, (k_title, k_val, k_col, k_tip) in enumerate(kpi_items):
    kpi_row.columnconfigure(col_idx, weight=1, uniform="kpi")
    tile = tk.Frame(
        kpi_row,
        bg=CARD_ELEVATED,
        padx=11,
        pady=6,
        highlightbackground=BORDER_SOFT,
        highlightthickness=1,
    )
    tile.grid(row=0, column=col_idx, sticky="ew", padx=(0, 8 if col_idx < len(kpi_items) - 1 else 0))
    l1 = tk.Label(tile, text=k_title, font=("Segoe UI", 7, "bold"), fg=TEXT_MUTED, bg=CARD_ELEVATED)
    l1.pack(anchor="w")
    l2 = tk.Label(tile, text=k_val, font=("Segoe UI", 10, "bold"), fg=k_col, bg=CARD_ELEVATED)
    l2.pack(anchor="w")

    def bind_tile_hover(w_tile, w_l1, w_l2, tip_text):
        def on_enter(_e):
            for w in (w_tile, w_l1, w_l2):
                w.configure(bg=CARD_HOVER)
            w_tile.configure(highlightbackground=VIOLET_ACCENT)
            w_l1.configure(fg=TEXT_BRIGHT)
            hover_bar.configure(text=tip_text, fg=TEXT_BRIGHT)

        def on_leave(_e):
            for w in (w_tile, w_l1, w_l2):
                w.configure(bg=CARD_ELEVATED)
            w_tile.configure(highlightbackground=BORDER_SOFT)
            w_l1.configure(fg=TEXT_MUTED)
            hover_bar.configure(text="Hover over KPI pills or chart bars for live details — enter your numbers or run a what-if check.", fg=TEXT_MUTED)

        for widget in (w_tile, w_l1, w_l2):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

    bind_tile_hover(tile, l1, l2, k_tip)

body = tk.Frame(outer, bg=BG_SLATE)
body.pack(fill=tk.BOTH, expand=True)

# left column: 4. entering new data + result banner + what-if shocks & run log
left_col = tk.Frame(body, bg=BG_SLATE, width=380)
left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
left_col.pack_propagate(False)

form_card = tk.Frame(
    left_col,
    bg=CARD_SLATE,
    padx=15,
    pady=12,
    highlightbackground=BORDER_SOFT,
    highlightthickness=1,
)
form_card.pack(fill=tk.X, pady=(0, 10))

tk.Label(form_card, text="YOUR LOAN PROFILE", font=("Segoe UI", 10, "bold"), fg=TEXT_BRIGHT, bg=CARD_SLATE).pack(anchor="w", pady=(0, 6))


def bind_entry_hover(entry_box):
    entry_box.bind("<Enter>", lambda _e: entry_box.configure(highlightbackground=VIOLET_ACCENT))
    entry_box.bind(
        "<Leave>",
        lambda _e: entry_box.configure(highlightbackground=BORDER_SOFT if root.focus_get() != entry_box else MINT_ACCENT),
    )
    entry_box.bind("<FocusIn>", lambda _e: entry_box.configure(highlightbackground=MINT_ACCENT))
    entry_box.bind("<FocusOut>", lambda _e: entry_box.configure(highlightbackground=BORDER_SOFT))


tk.Label(form_card, text="ANNUAL INCOME ($)", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=CARD_SLATE).pack(anchor="w")
income_entry = tk.Entry(
    form_card,
    font=("Segoe UI", 10),
    bg=INPUT_SLATE,
    fg=TEXT_BRIGHT,
    insertbackground=MINT_ACCENT,
    relief="flat",
    highlightbackground=BORDER_SOFT,
    highlightthickness=1,
)
income_entry.pack(fill=tk.X, ipady=5, pady=(2, 8))
income_entry.insert(0, "65000")
bind_entry_hover(income_entry)

tk.Label(form_card, text="DESIRED LOAN AMOUNT ($)", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=CARD_SLATE).pack(anchor="w")
loan_amount_entry = tk.Entry(
    form_card,
    font=("Segoe UI", 10),
    bg=INPUT_SLATE,
    fg=TEXT_BRIGHT,
    insertbackground=MINT_ACCENT,
    relief="flat",
    highlightbackground=BORDER_SOFT,
    highlightthickness=1,
)
loan_amount_entry.pack(fill=tk.X, ipady=5, pady=(2, 8))
loan_amount_entry.insert(0, "15000")
bind_entry_hover(loan_amount_entry)

tk.Label(form_card, text="CREDIT SCORE", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=CARD_SLATE).pack(anchor="w")
credit_score_entry = tk.Entry(
    form_card,
    font=("Segoe UI", 10),
    bg=INPUT_SLATE,
    fg=TEXT_BRIGHT,
    insertbackground=MINT_ACCENT,
    relief="flat",
    highlightbackground=BORDER_SOFT,
    highlightthickness=1,
)
credit_score_entry.pack(fill=tk.X, ipady=5, pady=(2, 10))
credit_score_entry.insert(0, "680")
bind_entry_hover(credit_score_entry)

result_box = tk.Frame(
    form_card,
    bg=CARD_ELEVATED,
    padx=12,
    pady=9,
    highlightbackground=BORDER_SOFT,
    highlightthickness=1,
)
result_box.pack(fill=tk.X, pady=(10, 0))

result_label = tk.Label(
    result_box,
    text="Status: Ready — click Predict Loan Status",
    font=("Segoe UI", 10, "bold"),
    fg=TEXT_BRIGHT,
    bg=CARD_ELEVATED,
    wraplength=315,
    justify="left",
)
result_label.pack(anchor="w")

sim_summary_label = tk.Label(
    result_box,
    text=f"Nearest-{best_k} neighbor votes & LTI ratio will show here.",
    font=("Segoe UI", 8),
    fg=TEXT_MUTED,
    bg=CARD_ELEVATED,
    wraplength=315,
    justify="left",
)
sim_summary_label.pack(anchor="w", pady=(2, 0))

# what-if shocks & run log card
sim_card = tk.Frame(
    left_col,
    bg=CARD_SLATE,
    padx=15,
    pady=11,
    highlightbackground=BORDER_SOFT,
    highlightthickness=1,
)
sim_card.pack(fill=tk.BOTH, expand=True)

log_header_row = tk.Frame(sim_card, bg=CARD_SLATE)
log_header_row.pack(fill=tk.X, pady=(0, 6))

tk.Label(log_header_row, text="WHAT-IF SHOCKS & RUN LOG", font=("Segoe UI", 9, "bold"), fg=VIOLET_ACCENT, bg=CARD_SLATE).pack(side=tk.LEFT)


def clear_run_log():
    sim_log.delete(0, tk.END)


clear_log_btn = tk.Button(
    log_header_row,
    text="Clear Log",
    font=("Segoe UI", 8, "bold"),
    fg=TEXT_MUTED,
    bg=CARD_ELEVATED,
    activebackground=CARD_HOVER,
    activeforeground=TEXT_BRIGHT,
    relief="flat",
    cursor="hand2",
    padx=8,
    pady=2,
    highlightbackground=BORDER_SOFT,
    highlightthickness=1,
    command=clear_run_log,
)
clear_log_btn.pack(side=tk.RIGHT)
clear_log_btn.bind("<Enter>", lambda _e: clear_log_btn.configure(bg=CARD_HOVER, fg=TEXT_BRIGHT, highlightbackground=AMBER_ACCENT))
clear_log_btn.bind("<Leave>", lambda _e: clear_log_btn.configure(bg=CARD_ELEVATED, fg=TEXT_MUTED, highlightbackground=BORDER_SOFT))

shock_row = tk.Frame(sim_card, bg=CARD_SLATE)
shock_row.pack(fill=tk.X, pady=(0, 8))

sim_log = tk.Listbox(
    sim_card,
    font=("Consolas", 8),
    bg=INPUT_SLATE,
    fg=TEXT_BRIGHT,
    selectbackground=CARD_HOVER,
    selectforeground=MINT_ACCENT,
    relief="flat",
    highlightbackground=BORDER_SOFT,
    highlightthickness=1,
    activestyle="none",
)
sim_log.pack(fill=tk.BOTH, expand=True)

# right column: typical loan applicant meter + SINGLE graph
right_col = tk.Frame(body, bg=BG_SLATE)
right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

typical_card = tk.Frame(
    right_col,
    bg=CARD_SLATE,
    padx=16,
    pady=11,
    highlightbackground=BORDER_SOFT,
    highlightthickness=1,
)
typical_card.pack(fill=tk.X, pady=(0, 10))

tk.Label(
    typical_card,
    text="TYPICAL LOAN APPLICANT METER  (AMBER TICK = TYPICAL APPLICANT)",
    font=("Segoe UI", 9, "bold"),
    fg=AMBER_ACCENT,
    bg=CARD_SLATE,
).pack(anchor="w", pady=(0, 5))

typical_canvas = tk.Canvas(typical_card, height=96, bg=CARD_ELEVATED, relief="flat", highlightthickness=0)
typical_canvas.pack(fill=tk.X)

last_income_holder = [0.0]
last_lti_holder = [0.0]
last_credit_holder = [0.0]
income_meter_max = max(avg_income * 2.0, 150000.0)


def draw_typical_meters(user_income=0.0, user_lti=0.0, user_credit=0.0):
    last_income_holder[0] = user_income
    last_lti_holder[0] = user_lti
    last_credit_holder[0] = user_credit
    typical_canvas.delete("all")
    w = typical_canvas.winfo_width()
    if w < 100:
        w = 820
    bx0 = 165
    bx1 = max(w - 190, bx0 + 80)
    span = max(bx1 - bx0, 60)

    # meter 1: annual income ($0 to income_meter_max, typical applicant tick)
    typical_canvas.create_text(10, 18, anchor="w", text="Annual Income", fill=TEXT_BRIGHT, font=("Segoe UI", 8, "bold"))
    draw_rounded_rect(typical_canvas, bx0, 11, bx1, 25, radius=6, fill=INPUT_SLATE, outline=BORDER_SOFT)
    u_inc_norm = max(0.0, min(user_income, income_meter_max)) / income_meter_max if user_income > 0 else 0.0
    t_inc_norm = max(0.0, min(avg_income, income_meter_max)) / income_meter_max
    if u_inc_norm > 0:
        draw_rounded_rect(typical_canvas, bx0 + 1, 12, max(bx0 + 8, bx0 + span * u_inc_norm), 24, radius=5, fill="#38bdf8", outline="")
    typical_canvas.create_line(bx0 + span * t_inc_norm, 8, bx0 + span * t_inc_norm, 28, fill=AMBER_ACCENT, width=3)
    typical_canvas.create_text(
        w - 10,
        18,
        anchor="e",
        text=f"You ${user_income:,.0f} | Typ ${avg_income:,.0f}",
        fill=TEXT_BRIGHT,
        font=("Segoe UI", 8, "bold"),
    )

    # meter 2: loan-to-income ratio (0% to 100%, typical applicant tick)
    typical_canvas.create_text(10, 48, anchor="w", text="Loan / Income", fill=TEXT_BRIGHT, font=("Segoe UI", 8, "bold"))
    draw_rounded_rect(typical_canvas, bx0, 41, bx1, 55, radius=6, fill=INPUT_SLATE, outline=BORDER_SOFT)
    u_lti_pos = bx0 + span * (max(0.0, min(user_lti, 100.0)) / 100.0)
    t_lti_pos = bx0 + span * (max(0.0, min(typical_lti_pct, 100.0)) / 100.0)
    if u_lti_pos > bx0 + 4:
        draw_rounded_rect(typical_canvas, bx0 + 1, 42, max(bx0 + 8, u_lti_pos), 54, radius=5, fill=MINT_ACCENT, outline="")
    typical_canvas.create_line(t_lti_pos, 38, t_lti_pos, 58, fill=AMBER_ACCENT, width=3)
    typical_canvas.create_text(
        w - 10,
        48,
        anchor="e",
        text=f"You {user_lti:.0f}% | Typ {typical_lti_pct:.0f}%",
        fill=TEXT_BRIGHT,
        font=("Segoe UI", 8, "bold"),
    )

    # meter 3: credit score (300 to 850, typical applicant tick)
    typical_canvas.create_text(10, 78, anchor="w", text="Credit (300-850)", fill=TEXT_BRIGHT, font=("Segoe UI", 8, "bold"))
    draw_rounded_rect(typical_canvas, bx0, 71, bx1, 85, radius=6, fill=INPUT_SLATE, outline=BORDER_SOFT)
    u_cr_norm = (max(300.0, min(user_credit, 850.0)) - 300.0) / 550.0 if user_credit > 0 else 0.0
    t_cr_norm = (max(300.0, min(avg_credit, 850.0)) - 300.0) / 550.0
    if u_cr_norm > 0:
        draw_rounded_rect(typical_canvas, bx0 + 1, 72, max(bx0 + 8, bx0 + span * u_cr_norm), 84, radius=5, fill=VIOLET_ACCENT, outline="")
    typical_canvas.create_line(bx0 + span * t_cr_norm, 68, bx0 + span * t_cr_norm, 88, fill=AMBER_ACCENT, width=3)
    typical_canvas.create_text(
        w - 10,
        78,
        anchor="e",
        text=f"You {user_credit:.0f} | Typ {avg_credit:.0f}",
        fill=TEXT_BRIGHT,
        font=("Segoe UI", 8, "bold"),
    )


# single graph card: what-if scenario approval score chart (25-run simulation removed)
chart_card = tk.Frame(
    right_col,
    bg=CARD_SLATE,
    padx=16,
    pady=12,
    highlightbackground=BORDER_SOFT,
    highlightthickness=1,
)
chart_card.pack(fill=tk.BOTH, expand=True)

sim_fig = Figure(figsize=(8.0, 4.4), dpi=100, facecolor=CARD_SLATE)
sim_ax = sim_fig.add_subplot(111)
sim_ax.set_facecolor(CARD_ELEVATED)

sim_bar_labels = ["Your Profile", "-20% Income", "+25% Loan", "+45 Credit"]
sim_scores = [0.0, 0.0, 0.0, 0.0]
sim_bars_ref = []
sim_tip_ref = [None]
current_chart_title = ["What-If Scenario Approval Score (%)"]


def render_sim_graph(values_list, chart_title=None):
    global sim_scores
    sim_scores = list(values_list)
    if chart_title is not None:
        current_chart_title[0] = chart_title

    sim_ax.clear()
    sim_ax.set_facecolor(CARD_ELEVATED)

    x_idx = np.arange(len(sim_bar_labels))
    bar_palette = [MINT_ACCENT, "#38bdf8", AMBER_ACCENT, VIOLET_ACCENT]
    bars = sim_ax.bar(x_idx, sim_scores, width=0.45, color=bar_palette)

    sim_bars_ref.clear()
    sim_bars_ref.extend(bars)

    for b, val in zip(bars, sim_scores):
        sim_ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 2.5,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=TEXT_BRIGHT,
        )

    sim_ax.set_ylim(0, 118)
    sim_ax.set_title(current_chart_title[0], fontsize=10, fontweight="bold", color=TEXT_BRIGHT, pad=10)
    sim_ax.set_ylabel("Approval Score (%)", fontsize=8.5, color=TEXT_MUTED)
    sim_ax.set_xticks(x_idx)
    sim_ax.set_xticklabels(sim_bar_labels, fontsize=8.5, color=TEXT_BRIGHT)
    sim_ax.tick_params(axis="y", labelsize=8.5, colors=TEXT_BRIGHT)
    sim_ax.spines["top"].set_visible(False)
    sim_ax.spines["right"].set_visible(False)
    sim_ax.spines["bottom"].set_color(BORDER_SOFT)
    sim_ax.spines["left"].set_color(BORDER_SOFT)
    sim_ax.grid(axis="y", linestyle="--", color=GRID_SLATE, alpha=0.75)

    tip = sim_ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.38", fc=CARD_SLATE, ec=MINT_ACCENT, lw=1.2),
        color=TEXT_BRIGHT,
        fontsize=8,
        fontweight="bold",
    )
    tip.set_visible(False)
    sim_tip_ref[0] = tip

    sim_fig.subplots_adjust(left=0.10, right=0.96, top=0.86, bottom=0.15)


def on_sim_hover(event):
    tip = sim_tip_ref[0]
    if tip is None:
        return
    if event.inaxes != sim_ax:
        if tip.get_visible():
            tip.set_visible(False)
            sim_canvas.draw_idle()
        return

    for bar, lbl, val in zip(sim_bars_ref, sim_bar_labels, sim_scores):
        contains, _ = bar.contains(event)
        if contains:
            tip.xy = (bar.get_x() + bar.get_width() / 2, bar.get_height())
            tip.set_text(f"{lbl}: {val:.0f}%")
            tip.set_visible(True)
            sim_canvas.draw_idle()
            return

    if tip.get_visible():
        tip.set_visible(False)
        sim_canvas.draw_idle()


render_sim_graph(sim_scores)
sim_canvas = FigureCanvasTkAgg(sim_fig, master=chart_card)
sim_canvas.mpl_connect("motion_notify_event", on_sim_hover)
sim_canvas.draw()
sim_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def parse_applicant_inputs():
    custom_income = float(income_entry.get().strip())
    custom_loan_amnt = float(loan_amount_entry.get().strip())
    custom_credit_score = float(credit_score_entry.get().strip())
    return custom_income, custom_loan_amnt, custom_credit_score


def score_candidate_pct(inc, loan, cred):
    scaled_pt = scaler.transform([[inc, loan, cred]])[0].tolist()
    pred = predict_knn_label(train_features, train_targets, scaled_pt, best_k)
    return (100.0 if pred == 1 else 0.0), pred


def set_entry_value(entry_widget, new_val):
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, str(round(new_val)))


def handle_predict(custom_point_override=None, note_prefix="Check"):
    try:
        if custom_point_override is None:
            custom_income, custom_loan_amnt, custom_credit_score = parse_applicant_inputs()
        else:
            custom_income, custom_loan_amnt, custom_credit_score = custom_point_override
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for income, loan amount, and credit score.")
        return

    custom_point = [custom_income, custom_loan_amnt, custom_credit_score]
    scaled_custom_point = scaler.transform([custom_point])[0].tolist()
    # predicting
    custom_prediction = predict_knn_label(train_features, train_targets, scaled_custom_point, best_k)
    readable_prediction = status_names.get(custom_prediction, f"Status {custom_prediction}")
    appr_votes, rej_votes = get_neighbor_vote_counts(train_features, train_targets, scaled_custom_point, best_k)

    lti_pct = (custom_loan_amnt / custom_income * 100.0) if custom_income > 0 else 0.0

    if custom_prediction == 1:
        result_box.configure(bg=MINT_TINT, highlightbackground=MINT_ACCENT)
        result_label.configure(text=f"RESULT: {readable_prediction}", fg=MINT_ACCENT, bg=MINT_TINT)
        sim_summary_label.configure(
            text=f"Neighbors ({best_k}): {appr_votes} Approved / {rej_votes} Rejected  •  LTI {lti_pct:.1f}%",
            fg="#a7f3d0",
            bg=MINT_TINT,
        )
    else:
        result_box.configure(bg=AMBER_TINT, highlightbackground=AMBER_ACCENT)
        result_label.configure(text=f"RESULT: {readable_prediction}", fg=AMBER_ACCENT, bg=AMBER_TINT)
        sim_summary_label.configure(
            text=f"Neighbors ({best_k}): {appr_votes} Approved / {rej_votes} Rejected  •  LTI {lti_pct:.1f}%",
            fg="#fde68a",
            bg=AMBER_TINT,
        )

    draw_typical_meters(custom_income, lti_pct, custom_credit_score)

    base_pct, _ = score_candidate_pct(custom_income, custom_loan_amnt, custom_credit_score)
    inc_down_pct, _ = score_candidate_pct(custom_income * 0.80, custom_loan_amnt, custom_credit_score)
    loan_up_pct, _ = score_candidate_pct(custom_income, custom_loan_amnt * 1.25, custom_credit_score)
    cred_up_pct, _ = score_candidate_pct(custom_income, custom_loan_amnt, min(850.0, custom_credit_score + 45.0))

    render_sim_graph(
        [base_pct, inc_down_pct, loan_up_pct, cred_up_pct],
        chart_title="What-If Scenario Approval Score (%)",
    )
    sim_canvas.draw()
    root.update_idletasks()

    tag = "APPR" if custom_prediction == 1 else "REJ "
    sim_log.insert(0, f"[{tag}] {note_prefix}: ${custom_income:,.0f} | ${custom_loan_amnt:,.0f} | Cr {custom_credit_score:.0f}")
    print(f"Predicted loan_status for income={custom_income}, loan_amnt={custom_loan_amnt}, credit_score={custom_credit_score}: {readable_prediction}")


def run_stress_shock(mode):
    try:
        inc, loan, cred = parse_applicant_inputs()
    except ValueError:
        messagebox.showerror("Enter Values First", "Please enter income, loan amount, and credit score before running a shock.")
        return

    if mode == "income_drop":
        shocked_inc = inc * 0.80
        set_entry_value(income_entry, shocked_inc)
        handle_predict((shocked_inc, loan, cred), note_prefix="-20% Inc")
    elif mode == "loan_hike":
        shocked_loan = loan * 1.25
        set_entry_value(loan_amount_entry, shocked_loan)
        handle_predict((inc, shocked_loan, cred), note_prefix="+25% Loan")
    elif mode == "credit_boost":
        shocked_cred = min(850.0, cred + 50.0)
        set_entry_value(credit_score_entry, shocked_cred)
        handle_predict((inc, loan, shocked_cred), note_prefix="+50 Cr")


shock_buttons = [
    ("-20% Income", "income_drop", VIOLET_ACCENT),
    ("+25% Loan", "loan_hike", AMBER_ACCENT),
    ("+50 Credit", "credit_boost", MINT_ACCENT),
]

for s_text, s_mode, s_fg in shock_buttons:
    sb = tk.Button(
        shock_row,
        text=s_text,
        font=("Segoe UI", 8, "bold"),
        fg=s_fg,
        bg=CARD_ELEVATED,
        activebackground=CARD_HOVER,
        activeforeground=TEXT_BRIGHT,
        relief="flat",
        cursor="hand2",
        padx=7,
        pady=4,
        highlightbackground=BORDER_SOFT,
        highlightthickness=1,
        command=lambda m=s_mode: run_stress_shock(m),
    )
    sb.pack(side=tk.LEFT, padx=(0, 6))
    sb.bind("<Enter>", lambda _e, b=sb, c=s_fg: b.configure(bg=CARD_HOVER, highlightbackground=c))
    sb.bind("<Leave>", lambda _e, b=sb: b.configure(bg=CARD_ELEVATED, highlightbackground=BORDER_SOFT))

predict_btn = tk.Button(
    form_card,
    text="Predict Loan Status",
    font=("Segoe UI", 10, "bold"),
    fg="#06281e",
    bg=MINT_ACCENT,
    activebackground=MINT_DEEP,
    activeforeground=TEXT_BRIGHT,
    relief="flat",
    cursor="hand2",
    pady=7,
    command=lambda: handle_predict(),
)
predict_btn.pack(fill=tk.X)
predict_btn.bind("<Enter>", lambda _e: predict_btn.configure(bg="#6ee7b7"))
predict_btn.bind("<Leave>", lambda _e: predict_btn.configure(bg=MINT_ACCENT))

root.update_idletasks()
draw_typical_meters(0.0, 0.0, 0.0)
typical_canvas.bind("<Configure>", lambda _e: draw_typical_meters(last_income_holder[0], last_lti_holder[0], last_credit_holder[0]))

root.protocol("WM_DELETE_WINDOW", root.destroy)
root.mainloop()
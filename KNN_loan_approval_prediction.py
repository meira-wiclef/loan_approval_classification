from collections import Counter  # counts how often each class label appears
import ctypes  # enables crisp DPI scaling on Windows
import sys  # checks the operating system
import tkinter as tk  # builds the desktop graphical interface
import numpy as np  # used for vector squaring
import pandas as pd  # used to extract csv
from sklearn.metrics import accuracy_score  # evaluates accuracy
from sklearn.preprocessing import StandardScaler  # standardizes features so larger scales don't dominate distance

# 1. loading and preparing the dataset
csv_filename = "loan_data.csv"
raw_dataset = pd.read_csv(csv_filename)  # this is a pandas dataframe

age_column = "person_age"  # feature 1
income_column = "person_income"  # feature 2
emp_exp_column = "person_emp_exp"  # feature 3
loan_amount_column = "loan_amnt"  # feature 4
interest_rate_column = "loan_int_rate"  # feature 5
credit_score_column = "credit_score"  # feature 6
target_column = "loan_status"  # the column we are trying to predict
feature_columns = [
    age_column,
    income_column,
    emp_exp_column,
    loan_amount_column,
    interest_rate_column,
    credit_score_column,
]  # this is a list of strings

cleaned_dataset = raw_dataset.copy()
for col in feature_columns + [target_column]:  # list concatenation, so we can clean all columns at once
    # converts each column to numbers and converts any invalid ones to NaN
    cleaned_dataset[col] = pd.to_numeric(cleaned_dataset[col], errors="coerce")

# dropping invalid rows and keeping a small sample of data
loan_dataset = cleaned_dataset.dropna(subset=feature_columns + [target_column]).head(700).sample(frac=1, random_state=42)
total_records = len(loan_dataset)

# splitting into training data and test data
training_count = int(0.8 * total_records)
training_records = loan_dataset.iloc[:training_count]
testing_records = loan_dataset.iloc[training_count:]

raw_train_features = training_records[feature_columns].values.tolist()  # 2D array of applicants
train_targets = training_records[target_column].astype(int).tolist()  # 1D array of loan statuses

raw_test_features = testing_records[feature_columns].values.tolist()
test_targets = testing_records[target_column].astype(int).tolist()

# training averages for comparing applicant inputs against the model's data
train_means = {
    col: float(training_records[col].mean())
    for col in feature_columns
}

# fitting StandardScaler on training features only to avoid data leakage
scaler = StandardScaler()
train_features = scaler.fit_transform(raw_train_features).tolist()
test_features = scaler.transform(raw_test_features).tolist()


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


def get_neighbor_vote_summary(train_features, train_targets, target_point, num_neighbors, predicted_label):
    neighbor_distances = []

    for index in range(len(train_features)):
        distance = calculate_euclidean_distance(target_point, train_features[index])
        actual_label = train_targets[index]
        neighbor_distances.append((distance, actual_label))

    neighbor_distances.sort(key=get_distance_from_record)
    nearest_labels = [label for _dist, label in neighbor_distances[:num_neighbors]]
    votes_for_pred = Counter(nearest_labels)[predicted_label]
    return f"{votes_for_pred}/{num_neighbors} nearest neighbors voted {predicted_label}"


status_names = {
    0: "Rejected (0)",
    1: "Approved (1)",
}

# 2. testing for best k
k_values = list(range(1, 15, 2))
accuracies: list[accuracy_score] = []

for k in k_values:
    predicted_targets = [
        predict_knn_label(train_features, train_targets, test_point, k)
        for test_point in test_features
    ]
    accuracy = accuracy_score(test_targets, predicted_targets)
    accuracies.append(accuracy)

best_k = k_values[accuracies.index(max(accuracies))]
print("Best k from elbow evaluation:", best_k)


def enable_sharp_display():
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


# rounded rectangle helper for tkinter canvas
def draw_rounded_rect(canvas, x1, y1, x2, y2, radius=20, **kwargs):
    r = min(radius, abs(x2 - x1) / 2, abs(y2 - y1) / 2)
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
    return canvas.create_polygon(points, smooth=True, splinesteps=24, **kwargs)


class RoundedLoanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN Applicant Evaluator")
        self.root.geometry("1360x820")
        self.root.minsize(1120, 720)
        self.root.configure(bg="#131722")

        # palette
        self.bg_main = "#131722"
        self.card_bg = "#1C2232"
        self.card_border = "#2B344B"
        self.input_bg = "#252D42"
        self.text_main = "#F3F6FC"
        self.text_muted = "#9AA6BF"
        self.accent_teal = "#2DD4BF"
        self.accent_teal_hover = "#5EEAD4"
        self.accent_violet = "#A78BFA"
        self.accent_amber = "#FBBF24"
        self.approve_color = "#34D399"
        self.reject_color = "#F87171"

        self.defaults = {
            age_column: "29",
            income_column: "68000",
            emp_exp_column: "5",
            loan_amount_column: "12000",
            interest_rate_column: "10.5",
            credit_score_column: "690",
        }

        self.field_specs = [
            (age_column, "Age (person_age)"),
            (income_column, "Income (person_income)"),
            (emp_exp_column, "Exp. Yrs (person_emp_exp)"),
            (loan_amount_column, "Loan Amt (loan_amnt)"),
            (interest_rate_column, "Rate % (loan_int_rate)"),
            (credit_score_column, "Credit (credit_score)"),
        ]

        self.entries = {}
        self.current_state = None
        self.btn_hovered = False
        self.term_months_var = tk.IntVar(value=36)
        self._build_ui()
        self.evaluate_applicant()

    def _build_ui(self):
        self.root.bind("<Return>", lambda _event: self.evaluate_applicant())

        # top rounded banner that stretches horizontally
        self.top_canvas = tk.Canvas(self.root, height=72, bg=self.bg_main, highlightthickness=0)
        self.top_canvas.pack(fill="x", padx=20, pady=(14, 12))
        self.top_canvas.bind("<Configure>", lambda _e: self._redraw_top_banner())

        # main expandable two-column layout
        body_frame = tk.Frame(self.root, bg=self.bg_main)
        body_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        body_frame.columnconfigure(0, weight=42, minsize=440)
        body_frame.columnconfigure(1, weight=58, minsize=620)
        body_frame.rowconfigure(0, weight=1)

        left_col = tk.Frame(body_frame, bg=self.bg_main)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_col.columnconfigure(0, weight=1)
        left_col.rowconfigure(0, weight=1)
        left_col.rowconfigure(1, weight=0)
        left_col.rowconfigure(2, weight=0)

        right_col = tk.Frame(body_frame, bg=self.bg_main)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right_col.columnconfigure(0, weight=1, uniform="vis_col")
        right_col.columnconfigure(1, weight=1, uniform="vis_col")
        right_col.rowconfigure(0, weight=42)
        right_col.rowconfigure(1, weight=58)

        # left column: input card, predict button, compact verdict card
        self.input_canvas = tk.Canvas(left_col, bg=self.bg_main, highlightthickness=0)
        self.input_canvas.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.input_canvas.bind("<Configure>", lambda _e: self._redraw_input_card())

        for col_name, _label_text in self.field_specs:
            entry = tk.Entry(
                self.input_canvas,
                bg=self.input_bg,
                fg=self.text_main,
                insertbackground=self.accent_teal,
                relief="flat",
                font=("Segoe UI", 10),
                justify="right",
            )
            entry.insert(0, self.defaults[col_name])
            self.entries[col_name] = entry

        self.btn_canvas = tk.Canvas(left_col, height=46, bg=self.bg_main, highlightthickness=0, cursor="hand2")
        self.btn_canvas.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.btn_canvas.bind("<Configure>", lambda _e: self._redraw_predict_button())
        self.btn_canvas.bind("<Enter>", self._on_btn_enter)
        self.btn_canvas.bind("<Leave>", self._on_btn_leave)
        self.btn_canvas.bind("<Button-1>", self._on_btn_click)

        self.verdict_canvas = tk.Canvas(left_col, height=94, bg=self.bg_main, highlightthickness=0)
        self.verdict_canvas.grid(row=2, column=0, sticky="ew")
        self.verdict_canvas.bind("<Configure>", lambda _e: self._redraw_verdict())

        # right column top row: credit score gauge + adjustable payoff meter
        self.credit_gauge = tk.Canvas(right_col, bg=self.bg_main, highlightthickness=0)
        self.credit_gauge.grid(row=0, column=0, sticky="nsew", padx=(0, 7), pady=(0, 10))
        self.credit_gauge.bind("<Configure>", lambda _e: self._redraw_visuals())

        self.payoff_canvas = tk.Canvas(right_col, bg=self.bg_main, highlightthickness=0)
        self.payoff_canvas.grid(row=0, column=1, sticky="nsew", padx=(7, 0), pady=(0, 10))
        self.payoff_canvas.bind("<Configure>", lambda _e: self._redraw_visuals())

        self.term_scale = tk.Scale(
            self.payoff_canvas,
            from_=12,
            to=60,
            resolution=6,
            orient="horizontal",
            variable=self.term_months_var,
            bg=self.card_bg,
            fg=self.text_main,
            troughcolor=self.input_bg,
            highlightthickness=0,
            activebackground=self.accent_amber,
            font=("Segoe UI", 8, "bold"),
            command=lambda _val: self.evaluate_applicant(),
        )

        # right column bottom row: income vs loan chart + model averages comparison chart
        self.income_loan_chart = tk.Canvas(right_col, bg=self.bg_main, highlightthickness=0)
        self.income_loan_chart.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        self.income_loan_chart.bind("<Configure>", lambda _e: self._redraw_visuals())

        self.averages_chart = tk.Canvas(right_col, bg=self.bg_main, highlightthickness=0)
        self.averages_chart.grid(row=1, column=1, sticky="nsew", padx=(7, 0))
        self.averages_chart.bind("<Configure>", lambda _e: self._redraw_visuals())

    def _on_btn_enter(self, _event):
        self.btn_hovered = True
        self._redraw_predict_button()

    def _on_btn_leave(self, _event):
        self.btn_hovered = False
        self._redraw_predict_button()

    def _on_btn_click(self, _event):
        self.root.focus_set()
        self.evaluate_applicant()

    def _redraw_top_banner(self):
        c = self.top_canvas
        c.delete("all")
        w = max(c.winfo_width(), 600)
        h = max(c.winfo_height(), 60)

        draw_rounded_rect(c, 2, 2, w - 2, h - 2, radius=22, fill=self.card_bg, outline=self.card_border, width=1)
        c.create_text(
            26, h * 0.36, anchor="w",
            text="KNN Loan Applicant Evaluator",
            fill=self.text_main, font=("Segoe UI", 15, "bold")
        )
        c.create_text(
            26, h * 0.69, anchor="w",
            text=f"Sample: {total_records} records   |   Optimal k: {best_k}   |   6 Numeric Features",
            fill=self.text_muted, font=("Segoe UI", 10)
        )

        badge_w = 215
        badge_x1 = w - badge_w - 22
        badge_x2 = w - 22
        badge_y1 = 15
        badge_y2 = h - 15
        draw_rounded_rect(c, badge_x1, badge_y1, badge_x2, badge_y2, radius=16, fill="#163334", outline=self.accent_teal, width=1)
        c.create_text(
            (badge_x1 + badge_x2) / 2, h / 2,
            text=f"Accuracy: {accuracies.index(max(accuracies)):.3f}%",
            fill=self.accent_teal, font=("Segoe UI", 11, "bold")
        )

    def _redraw_input_card(self):
        c = self.input_canvas
        c.delete("all")
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 280)

        draw_rounded_rect(c, 2, 2, w - 2, h - 2, radius=24, fill=self.card_bg, outline=self.card_border, width=1)
        c.create_text(24, 26, anchor="w", text="Applicant Details", fill=self.text_main, font=("Segoe UI", 13, "bold"))

        num_fields = len(self.field_specs)
        top_offset = 60
        bottom_padding = 22
        row_step = max(36, (h - top_offset - bottom_padding) / num_fields)
        box_h = min(28, row_step - 9)
        box_w = min(145, max(110, int(w * 0.28)))
        box_x2 = w - 24
        box_x1 = box_x2 - box_w

        for idx, (col_name, label_text) in enumerate(self.field_specs):
            y_pos = top_offset + idx * row_step
            c.create_text(
                24, y_pos + box_h / 2, anchor="w",
                text=label_text, fill=self.text_muted, font=("Segoe UI", 9, "bold")
            )
            draw_rounded_rect(c, box_x1, y_pos, box_x2, y_pos + box_h, radius=10, fill=self.input_bg, outline="#384361", width=1)
            entry = self.entries[col_name]
            entry.place(x=box_x1 + 8, y=y_pos + 3, width=box_w - 16, height=max(16, box_h - 6))

    def _redraw_predict_button(self):
        c = self.btn_canvas
        c.delete("all")
        w = max(c.winfo_width(), 200)
        h = max(c.winfo_height(), 40)

        fill_col = self.accent_teal_hover if self.btn_hovered else self.accent_teal
        draw_rounded_rect(c, 2, 2, w - 2, h - 2, radius=20, fill=fill_col, outline="")
        c.create_text(w / 2, h / 2, text="Predict Loan Status", fill="#0B131E", font=("Segoe UI", 11, "bold"))

    def _redraw_verdict(self):
        c = self.verdict_canvas
        c.delete("all")
        w = max(c.winfo_width(), 240)
        h = max(c.winfo_height(), 86)

        if self.current_state is None:
            status_title, subtitle, approved = "READY", "Enter values and click Predict", None
        else:
            status_title, subtitle, approved = self.current_state["verdict"]

        if approved is True:
            border = self.approve_color
            badge_bg = "#13332A"
            title_col = self.approve_color
        elif approved is False:
            border = self.reject_color
            badge_bg = "#361920"
            title_col = self.reject_color
        else:
            border = self.card_border
            badge_bg = self.card_bg
            title_col = self.text_main

        draw_rounded_rect(c, 2, 2, w - 2, h - 2, radius=20, fill=self.card_bg, outline=border, width=2)
        badge_size = min(64, h - 24)
        bx1 = 16
        by1 = (h - badge_size) / 2
        bx2 = bx1 + badge_size
        by2 = by1 + badge_size
        draw_rounded_rect(c, bx1, by1, bx2, by2, radius=14, fill=badge_bg, outline=border, width=1)
        c.create_text(
            (bx1 + bx2) / 2, h / 2,
            text="1" if approved is True else ("0" if approved is False else "?"),
            fill=title_col, font=("Segoe UI", 18, "bold")
        )
        text_x = bx2 + 14
        c.create_text(text_x, h * 0.36, anchor="w", text=status_title, fill=title_col, font=("Segoe UI", 13, "bold"))
        c.create_text(text_x, h * 0.66, anchor="w", text=subtitle, fill=self.text_muted, font=("Segoe UI", 9))

    def _redraw_visuals(self):
        if self.current_state is None:
            return
        st = self.current_state
        self._draw_circular_gauge(
            self.credit_gauge,
            "Credit Score Gauge",
            st["credit_text"],
            st["credit_ratio"],
            self.accent_teal,
        )
        self._draw_payoff_meter(
            self.payoff_canvas,
            st["approved"],
            st["term_months"],
            st["monthly_payment"],
            st["total_repayment"],
            st["interest_rate"],
        )
        self._draw_income_loan_chart(self.income_loan_chart, st["income"], st["loan_amnt"])
        self._draw_averages_chart(self.averages_chart, st["scale_rows"])

    def _draw_circular_gauge(self, canvas, title, value_text, ratio, ring_color):
        canvas.delete("all")
        w = max(canvas.winfo_width(), 220)
        h = max(canvas.winfo_height(), 160)

        draw_rounded_rect(canvas, 2, 2, w - 2, h - 2, radius=24, fill=self.card_bg, outline=self.card_border, width=1)
        canvas.create_text(22, 24, anchor="w", text=title, fill=self.text_main, font=("Segoe UI", 11, "bold"))

        clamped = max(0.0, min(1.0, ratio))
        cx = w / 2
        cy = 34 + (h - 46) / 2
        r = max(36, min((w - 64) / 2, (h - 72) / 2, 66))
        x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r

        # background circular track
        canvas.create_oval(x0, y0, x1, y1, outline="#2B354C", width=11)
        # active circular arc starting at top (90 deg, sweeping clockwise)
        if clamped > 0:
            extent = -359.9 * clamped
            canvas.create_arc(x0, y0, x1, y1, start=90, extent=extent, style="arc", outline=ring_color, width=11)

        canvas.create_text(cx, cy - 6, text=value_text, fill=self.text_main, font=("Segoe UI", 12, "bold"))
        canvas.create_text(cx, cy + 14, text=f"{round(clamped * 100)}%", fill=self.text_muted, font=("Segoe UI", 9))

    def _draw_payoff_meter(self, canvas, approved, term_months, monthly_payment, total_repayment, rate_pct):
        canvas.delete("all")
        w = max(canvas.winfo_width(), 240)
        h = max(canvas.winfo_height(), 160)

        draw_rounded_rect(canvas, 2, 2, w - 2, h - 2, radius=24, fill=self.card_bg, outline=self.card_border, width=1)
        canvas.create_text(22, 24, anchor="w", text="Approved Payoff Meter", fill=self.text_main, font=("Segoe UI", 11, "bold"))

        canvas.create_text(
            22, 52, anchor="w",
            text=f"Repayment Term: {term_months} mos  (Rate: {rate_pct:g}%)",
            fill=self.text_muted, font=("Segoe UI", 9, "bold")
        )

        # place adjustable slider inside the card
        scale_w = max(140, int(w - 44))
        self.term_scale.place(x=22, y=68, width=scale_w, height=40)

        info_y = 124
        if approved:
            canvas.create_text(
                22, info_y, anchor="w",
                text=f"Est. Total: ${total_repayment:,.0f}   |   Monthly: ${monthly_payment:,.0f}/mo",
                fill=self.approve_color, font=("Segoe UI", 10, "bold")
            )
            canvas.create_text(
                22, info_y + 20, anchor="w",
                text="Slide term (12–60 months) to preview interest cost if approved",
                fill=self.text_muted, font=("Segoe UI", 8)
            )
        else:
            canvas.create_text(
                22, info_y, anchor="w",
                text="Loan not approved — slider still previews hypothetical cost",
                fill=self.reject_color, font=("Segoe UI", 9, "bold")
            )
            canvas.create_text(
                22, info_y + 20, anchor="w",
                text=f"If approved at {rate_pct:g}% over {term_months}m: ${total_repayment:,.0f} (${monthly_payment:,.0f}/mo)",
                fill=self.text_muted, font=("Segoe UI", 8)
            )

    def _draw_income_loan_chart(self, canvas, income, loan_amnt):
        canvas.delete("all")
        w = max(canvas.winfo_width(), 220)
        h = max(canvas.winfo_height(), 180)

        draw_rounded_rect(canvas, 2, 2, w - 2, h - 2, radius=24, fill=self.card_bg, outline=self.card_border, width=1)
        canvas.create_text(22, 24, anchor="w", text="Income vs. Loan Chart", fill=self.text_main, font=("Segoe UI", 11, "bold"))

        base_y = h - 38
        max_bar_h = max(44, h - 102)
        max_val = max(income, loan_amnt, 1.0)
        inc_h = max(10, int((income / max_val) * max_bar_h))
        loan_h = max(10, int((loan_amnt / max_val) * max_bar_h))

        # baseline
        canvas.create_line(36, base_y, w - 36, base_y, fill="#384361", width=2)

        bar_w = min(68, max(42, int(w * 0.16)))
        inc_cx = w * 0.32
        loan_cx = w * 0.68

        # income bar
        draw_rounded_rect(canvas, inc_cx - bar_w / 2, base_y - inc_h, inc_cx + bar_w / 2, base_y, radius=10, fill=self.accent_teal, outline="")
        canvas.create_text(inc_cx, base_y + 16, text="Income", fill=self.text_muted, font=("Segoe UI", 9))
        canvas.create_text(inc_cx, base_y - inc_h - 11, text=f"{income:,.0f}", fill=self.text_main, font=("Segoe UI", 9, "bold"))

        # loan bar
        draw_rounded_rect(canvas, loan_cx - bar_w / 2, base_y - loan_h, loan_cx + bar_w / 2, base_y, radius=10, fill=self.accent_violet, outline="")
        canvas.create_text(loan_cx, base_y + 16, text="Loan", fill=self.text_muted, font=("Segoe UI", 9))
        canvas.create_text(loan_cx, base_y - loan_h - 11, text=f"{loan_amnt:,.0f}", fill=self.text_main, font=("Segoe UI", 9, "bold"))

    def _draw_averages_chart(self, canvas, scale_rows):
        canvas.delete("all")
        w = max(canvas.winfo_width(), 240)
        h = max(canvas.winfo_height(), 180)

        draw_rounded_rect(canvas, 2, 2, w - 2, h - 2, radius=24, fill=self.card_bg, outline=self.card_border, width=1)
        canvas.create_text(22, 24, anchor="w", text="Applicant vs. Model Avg (Scaled)", fill=self.text_main, font=("Segoe UI", 11, "bold"))

        # legend
        canvas.create_rectangle(w - 155, 19, w - 143, 29, fill=self.accent_amber, outline="")
        canvas.create_text(w - 138, 24, anchor="w", text="Applicant", fill=self.text_muted, font=("Segoe UI", 8))
        canvas.create_rectangle(w - 82, 19, w - 70, 29, fill=self.accent_violet, outline="")
        canvas.create_text(w - 65, 24, anchor="w", text="Avg", fill=self.text_muted, font=("Segoe UI", 8))

        if not scale_rows:
            return

        left_x = 22
        right_x = w - 22
        bar_area_x1 = 118
        bar_area_x2 = right_x - 8
        bar_span = max(40, bar_area_x2 - bar_area_x1)

        top_y = 48
        row_h = max(26, (h - top_y - 16) / len(scale_rows))

        for idx, (label, user_val, avg_val, max_scale) in enumerate(scale_rows):
            y_mid = top_y + idx * row_h + row_h / 2
            canvas.create_text(left_x, y_mid, anchor="w", text=label, fill=self.text_muted, font=("Segoe UI", 8, "bold"))

            u_ratio = max(0.0, min(1.0, user_val / max_scale))
            a_ratio = max(0.0, min(1.0, avg_val / max_scale))

            # applicant bar (top half of row)
            canvas.create_rectangle(bar_area_x1, y_mid - 8, bar_area_x2, y_mid - 1, fill="#283148", outline="")
            u_end = max(bar_area_x1 + 4, bar_area_x1 + u_ratio * bar_span)
            canvas.create_rectangle(bar_area_x1, y_mid - 8, u_end, y_mid - 1, fill=self.accent_amber, outline="")

            # training average bar (bottom half of row)
            canvas.create_rectangle(bar_area_x1, y_mid + 1, bar_area_x2, y_mid + 8, fill="#283148", outline="")
            a_end = max(bar_area_x1 + 4, bar_area_x1 + a_ratio * bar_span)
            canvas.create_rectangle(bar_area_x1, y_mid + 1, a_end, y_mid + 8, fill=self.accent_violet, outline="")

    def evaluate_applicant(self):
        term_months = max(1, int(self.term_months_var.get()))
        try:
            custom_age = float(self.entries[age_column].get())
            custom_income = float(self.entries[income_column].get())
            custom_emp_exp = float(self.entries[emp_exp_column].get())
            custom_loan_amnt = float(self.entries[loan_amount_column].get())
            custom_interest_rate = float(self.entries[interest_rate_column].get())
            custom_credit_score = float(self.entries[credit_score_column].get())
        except ValueError:
            self.current_state = {
                "verdict": ("INVALID INPUT", "Enter valid numbers in all 6 fields", None),
                "approved": False,
                "credit_text": "--",
                "credit_ratio": 0.0,
                "term_months": term_months,
                "monthly_payment": 0.0,
                "total_repayment": 0.0,
                "interest_rate": 0.0,
                "income": 0.0,
                "loan_amnt": 0.0,
                "scale_rows": [],
            }
            self._redraw_verdict()
            self._redraw_visuals()
            return

        custom_point = [
            custom_age,
            custom_income,
            custom_emp_exp,
            custom_loan_amnt,
            custom_interest_rate,
            custom_credit_score,
        ]
        scaled_custom_point = scaler.transform([custom_point])[0].tolist()
        # predicting
        custom_prediction = predict_knn_label(train_features, train_targets, scaled_custom_point, best_k)
        readable_prediction = status_names.get(custom_prediction, f"Status {custom_prediction}")
        vote_summary = get_neighbor_vote_summary(
            train_features, train_targets, scaled_custom_point, best_k, custom_prediction
        )
        is_approved = custom_prediction == 1

        # standard amortization formula using the interest rate and selected term
        monthly_rate = (custom_interest_rate / 100.0) / 12.0
        if custom_loan_amnt <= 0:
            monthly_payment = 0.0
            total_repayment = 0.0
        elif monthly_rate > 0:
            factor = (1.0 + monthly_rate) ** term_months
            monthly_payment = custom_loan_amnt * (monthly_rate * factor) / (factor - 1.0)
            total_repayment = monthly_payment * term_months
        else:
            monthly_payment = custom_loan_amnt / term_months
            total_repayment = custom_loan_amnt

        score_ratio = max(0.0, min(1.0, (custom_credit_score - 300) / 550.0))

        scale_rows = [
            ("Age", custom_age, train_means[age_column], 70.0),
            ("Income", custom_income, train_means[income_column], 150000.0),
            ("Exp. Yrs", custom_emp_exp, train_means[emp_exp_column], 30.0),
            ("Loan Amt", custom_loan_amnt, train_means[loan_amount_column], 40000.0),
            ("Rate %", custom_interest_rate, train_means[interest_rate_column], 25.0),
            ("Credit", custom_credit_score, train_means[credit_score_column], 850.0),
        ]

        self.current_state = {
            "verdict": (
                readable_prediction,
                vote_summary,
                is_approved,
            ),
            "approved": is_approved,
            "credit_text": f"{custom_credit_score:g}",
            "credit_ratio": score_ratio,
            "term_months": term_months,
            "monthly_payment": monthly_payment,
            "total_repayment": total_repayment,
            "interest_rate": custom_interest_rate,
            "income": custom_income,
            "loan_amnt": custom_loan_amnt,
            "scale_rows": scale_rows,
        }

        self._redraw_verdict()
        self._redraw_visuals()


if __name__ == "__main__":
    enable_sharp_display()
    app_root = tk.Tk()
    RoundedLoanApp(app_root)
    app_root.mainloop()
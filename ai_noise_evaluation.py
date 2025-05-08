import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from fpdf import FPDF
import tempfile

st.title("Noisy ANN Trainer")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Select inputs and outputs
    columns = df.columns.tolist()
    if columns:
        input_cols = st.multiselect("Select input features", options=columns, default=[])
        output_cols = st.multiselect("Select output columns", options=columns, default=[])

        if input_cols and output_cols:
            # Noise level
            noise_level = st.slider("Select noise level (0% - 100%)", 0, 100, 10)

            # Run button
            if st.button("Train ANN"):
                # Prepare data
                X = df[input_cols].values
                y = df[output_cols].values

                # Split clean data first
                X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Add white noise (relative to signal magnitude)
                X_train = X_train_clean + np.random.normal(0, noise_level / 100.0, X_train_clean.shape) * X_train_clean
                X_test = X_test_clean + np.random.normal(0, noise_level / 100.0, X_test_clean.shape) * X_test_clean

                # Scale
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_train = scaler_X.fit_transform(X_train)
                X_test = scaler_X.transform(X_test)
                y_train = scaler_y.fit_transform(y_train_clean)
                y_test = scaler_y.transform(y_test_clean)

                # Build shallow ANN
                model = Sequential([
                    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
                    Dense(len(output_cols))
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=100, verbose=0)

                # Predict
                y_pred = model.predict(X_test)
                y_pred_inv = scaler_y.inverse_transform(y_pred)
                y_test_inv = scaler_y.inverse_transform(y_test)

                # R² Score
                r2_scores = [r2_score(y_test_inv[:, i], y_pred_inv[:, i]) for i in range(len(output_cols))]
                for i, col in enumerate(output_cols):
                    st.metric(f"R² Score for {col}", round(r2_scores[i], 4))

                # Plot actual vs predicted for each output (test set only)
                for i, col in enumerate(output_cols):
                    fig, ax = plt.subplots()
                    ax.plot(y_test_clean[:, i], label='Original Clean (Test)', linestyle='--', marker='o', markersize=4)
                    ax.plot(y_test_inv[:, i], label='Original Noisy (Test)')
                    ax.plot(y_pred_inv[:, i], label='Predicted')
                    ax.legend()
                    ax.set_title(f"Original vs Predicted - {col}")
                    st.pyplot(fig)

                # Plot noisy vs clean inputs for each feature
                noise = np.random.normal(0, noise_level / 100.0, X.shape) * X
                X_noisy_full = X + noise
                for i, col in enumerate(input_cols):
                    fig2, ax2 = plt.subplots()
                    ax2.plot(X[:, i], label=f'Original Input ({col})')
                    ax2.plot(X_noisy_full[:, i], label=f'Noisy Input ({col})', alpha=0.7)
                    ax2.legend()
                    ax2.set_title(f"Original vs Noisy Input - {col}")
                    st.pyplot(fig2)

                # Generate PDF
                class PDF(FPDF):
                    def header(self):
                        self.set_font('Arial', 'B', 12)
                        self.cell(0, 10, 'ANN Training Report', 0, 1, 'C')

                    def section_title(self, title):
                        self.set_font('Arial', 'B', 10)
                        self.cell(0, 10, title, 0, 1)

                    def section_body(self, body):
                        self.set_font('Arial', '', 10)
                        self.multi_cell(0, 10, body)

                pdf = PDF()
                pdf.add_page()
                pdf.section_title("Input Features")
                pdf.section_body(", ".join(input_cols))
                pdf.section_title("Output Features")
                pdf.section_body(", ".join(output_cols))
                pdf.section_title("Noise Level")
                pdf.section_body(f"{noise_level}%")
                pdf.section_title("R² Scores")
                for i, col in enumerate(output_cols):
                    pdf.section_body(f"{col}: {round(r2_scores[i], 4)}")

                # Save output plots temporarily and insert in PDF
                for i, col in enumerate(output_cols):
                    fig, ax = plt.subplots()
                    ax.plot(y_test_clean[:, i], label='Original Clean (Test)', linestyle='--', marker='o', markersize=4)
                    ax.plot(y_test_inv[:, i], label='Original Noisy (Test)')
                    ax.plot(y_pred_inv[:, i], label='Predicted')
                    ax.legend()
                    ax.set_title(f"Original vs Predicted - {col}")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        fig.savefig(tmpfile.name)
                        pdf.image(tmpfile.name, w=180)

                for i, col in enumerate(input_cols):
                    fig2, ax2 = plt.subplots()
                    ax2.plot(X[:, i], label=f'Original Input ({col})')
                    ax2.plot(X_noisy_full[:, i], label=f'Noisy Input ({col})', alpha=0.7)
                    ax2.legend()
                    ax2.set_title(f"Original vs Noisy Input - {col}")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile2:
                        fig2.savefig(tmpfile2.name)
                        pdf.image(tmpfile2.name, w=180)

                pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                pdf.output(pdf_output.name)

                with open(pdf_output.name, "rb") as f:
                    st.download_button("Download PDF Report", f, file_name="ann_report.pdf")


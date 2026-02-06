import json
import os

target_file = "Daniella Han Xue En_2404908B.ipynb"

if not os.path.exists(target_file):
    print(f"Error: File {target_file} not found.")
    exit(1)

with open(target_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# --- Define New Cells ---

# 1. Descriptive Stats
cell_desc = {
   "cell_type": "code",
   "execution_count": None,
   "id": "eda_distribution_desc",
   "metadata": {"scrolled": True},
   "outputs": [],
   "source": [
    "print(\"\\nDescriptive statistics:\")\n",
    "display(df[numerical_cols].describe())\n"
   ]
}

# 2. Histograms
cell_hist = {
   "cell_type": "code",
   "execution_count": None,
   "id": "eda_distribution_hist",
   "metadata": {"scrolled": True},
   "outputs": [],
   "source": [
    "key_numerical = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', \n",
    "                 'num_medications', 'number_outpatient', 'number_emergency', \n",
    "                 'number_inpatient', 'number_diagnoses']\n",
    "\n",
    "fig, axes = plt.subplots(2, 4, figsize=(18, 8))\n",
    "axes = axes.ravel()\n",
    "\n",
    "for idx, col in enumerate(key_numerical):\n",
    "    if col in df.columns:\n",
    "        axes[idx].hist(df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)\n",
    "        axes[idx].set_title(f'{col}', fontweight='bold')\n",
    "        axes[idx].set_xlabel('Value')\n",
    "        axes[idx].set_ylabel('Frequency')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
}

# 3. Categorical Summary
cell_cat = {
   "cell_type": "code",
   "execution_count": None,
   "id": "eda_distribution_cat",
   "metadata": {"scrolled": True},
   "outputs": [],
   "source": [
    "print(\"\\n[4.2] CATEGORICAL FEATURES SUMMARY\")\n",
    "print(\"-\" * 80)\n",
    "\n",
    "categorical_cols = df.select_dtypes(include=['object']).columns.tolist()\n",
    "print(f\"Categorical columns: {len(categorical_cols)}\")\n",
    "\n",
    "key_categorical = ['age', 'gender', 'race', 'admission_type_id', 'discharge_disposition_id']\n",
    "\n",
    "for col in key_categorical:\n",
    "    if col in df.columns:\n",
    "        print(f\"\\n{col}: {df[col].nunique()} unique values\")\n",
    "        print(df[col].value_counts().head())\n"
   ]
}

# 4. Insight Analysis - Distributions
cell_insight_dist = {
   "cell_type": "markdown",
   "id": "eda_insight_dist",
   "metadata": {},
   "source": [
    "#### Insightful Analysis of Feature Distributions\n",
    "\n",
    "> [!NOTE]\n",
    "> Understanding data distribution is critical for model selection and feature engineering.\n",
    "\n",
    "1.  **Numerical Features:**\n",
    "    *   **Skewness**: Features like `number_emergency`, `number_inpatient`, and `number_outpatient` are highly right-skewed with a large number of zeros. This indicates that most patients do not have prior visits, making these variables sparse. Transformations (e.g., log-transform) may be beneficial.\n",
    "    *   **Time in Hospital**: The distribution is right-skewed, with the majority of stays being short (1-4 days). This mirrors typical acute care patterns.\n",
    "    *   **Lab Procedures**: `num_lab_procedures` approximates a normal distribution but is fairly broad, reflecting the variability in patient complexity.\n",
    "\n",
    "2.  **Categorical Features:**\n",
    "    *   **Age Profile**: The dataset is dominated by older patients (buckets `[60-70)`, `[70-80)`), which aligns with the higher prevalence of diabetes and complications in geriatric populations.\n",
    "    *   **Readmission Balance**: Checking the target `readmitted` distribution (not shown here but implied) is crucial; class imbalance will require techniques like SMOTE or class-weighting during modeling."
   ]
}

# 5. Insight Analysis - Relationships
cell_insight_rel = {
   "cell_type": "markdown",
   "id": "eda_insight_rel",
   "metadata": {},
   "source": [
    "#### Insightful Analysis: Feature Impact on Readmission\n",
    "\n",
    "> [!IMPORTANT]\n",
    "> Key differentiators identified for readmitted vs. non-readmitted patients.\n",
    "\n",
    "*   **Inpatient Utilization**: The boxplots reveal a clear signal: patients readmitted within 30 days (`YES`) tend to have a **higher median number of prior inpatient visits** compared to those not readmitted (`NO`). This suggests that history of hospitalization is a strong predictor of future readmission risk.\n",
    "*   **Diagnoses Count**: There is a positive relationship between `number_diagnoses` and readmission. Patients with more recorded diagnoses (indicating higher medical complexity) are more prone to readmission.\n",
    "*   **Medication Volume**: `num_medications` shows a slight increase for readmitted patients, likely serving as a proxy for disease severity.\n",
    "*   **Implication**: Feature engineering should focus on capturing *intensity* of care (e.g., total visits, total diagnoses) as these show stronger separation than simple demographic factors."
   ]
}

# --- Action 1: Insert Stats, Hist, Cat, Insight after Header ---
# Header ID: 5e9a3421 "Understanding distribution..."
idx_header = -1
for i, c in enumerate(cells):
    if c.get('id') == "5e9a3421":
        idx_header = i
        break

if idx_header != -1:
    print(f"Found distribution header at index {idx_header}")
    # Insert in reverse order to keep correct sequence
    cells.insert(idx_header + 1, cell_insight_dist)
    cells.insert(idx_header + 1, cell_cat)
    cells.insert(idx_header + 1, cell_hist)
    cells.insert(idx_header + 1, cell_desc)
    print("Inserted Distribution content.")
else:
    print("Warning: Distribution header (5e9a3421) not found.")

# --- Action 2: Insert Relationship Insight after Plot ---
# Plot ID: migrated_eda_1
idx_plot = -1
for i, c in enumerate(cells):
    if c.get('id') == "migrated_eda_1":
        idx_plot = i
        break

if idx_plot != -1:
    print(f"Found Feature vs Target plot at index {idx_plot}")
    cells.insert(idx_plot + 1, cell_insight_rel)
    print("Inserted Relationship Insight.")
else:
    print("Warning: Feature vs Target plot (migrated_eda_1) not found.")

# --- Save ---
nb['cells'] = cells
with open(target_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Migration V2 Complete.")

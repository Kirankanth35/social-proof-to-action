<img width="2001" height="1125" alt="slide1_preview" src="https://github.com/user-attachments/assets/16745493-ca1b-4d37-b058-be2de63660b0" />

# social-proof-to-action
Graduate research project on algorithmic amplification, influencer authority, and high-effort social media engagement
This repository is a professionally organized GitHub version of a graduate analytics project centered on **social media performance, algorithmic visibility, influencer authority, and meaningful engagement**. It was assembled from the original project bundle and the final presentation deck, then cleaned into a publishable repository structure.

### Core idea
On algorithmic platforms, being widely seen is **not** the same as driving meaningful action. This project separates **early persuasive signals**, **algorithmic amplification**, **influencer authority**, and **high-effort engagement** to explain why some posts convert visibility into action while others do not.

---

## Repository at a glance

- **Project title:** From Social Proof to Action  
- **Program:** M.S. in Business Analytics, California State University, Northridge  
- **Primary deliverable:** `docs/presentations/CSUN_BANA698_Final_Presentation_Social_Proof_to_Action.pptx`  
- **Dataset:** synthetic instructional workbook with `10,000` posts, `35` variables, and `800` influencers  
- **Primary language:** Python  
- **Analysis style:** observational, explanatory, regression-based analytics

---

## Research questions

1. **Mechanism + authority** — Do early persuasive signals translate into broader visibility, and does authority shape that pathway?
2. **Diminishing returns** — At what point does additional reach add less meaningful engagement?
3. **High-visibility condition** — Does very high visibility weaken the predictive role of persuasive power?

---

## Data and scope

The repository includes the project workbook used in the analysis:

- **Posts:** 10,000
- **Variables:** 35
- **Unique influencers:** 800
- **Date range:** 2024-01-01 to 2025-12-30
- **Content types:** experiential=4,067, informational=3,466, promotional=2,467
- **Post formats:** image=4,526, video=3,481, carousel=1,993
- **Verified accounts:** 23.9%
- **Very high reach posts:** 5.0%

The canonical raw dataset is stored in:

- `data/raw/dataset_algorithmic_persuasion_10000.xlsx`

A processed feature-engineered file is stored in:

- `data/processed/dataset_feature_engineered.csv`

See also:

- `data/codebook/dataset_codebook.csv`
- `docs/summaries/dataset_profile.md`
- `docs/summaries/variable_dictionary.md`

---

## Methodology

The repository preserves the original analysis pipeline and outputs. The main workflow combines:

- feature engineering and variable transformation
- descriptive diagnostics and distribution plots
- clustered OLS for mediation and moderation analysis
- nonlinear reach modeling for diminishing returns
- high-visibility interaction testing
- consolidated hypothesis reporting

Primary scripts:

- `src/feature_engineering.py`
- `src/descriptive_statistics.py`
- `src/rq1_mediation_moderation.py`
- `src/hypothesis_tests_rq1_rq2_rq3.py`

Legacy exploratory scripts from the original bundle are preserved in `src/legacy/`.

---

## Key findings reflected in the outputs

The outputs in this repository support five central findings:

1. **Early persuasive signals are positively associated with algorithmic amplification.**
2. **The direct amplification → high-effort engagement relationship is weak on average**, which is why interpretation should focus on conditional pathways rather than a single uniform direct effect.
3. **Authority strengthens the pathway.** Conditional indirect effects are weakest at low authority and strongest at high authority.
4. **Reach shows diminishing returns.** The quadratic model implies a turning point around **log reach ≈ 14.18**, which corresponds to approximately **1,444,847 reach**.
5. **Under very high visibility, persuasive power becomes less predictive** of meaningful engagement.

Selected visual outputs:

<img width="646" height="350" alt="image" src="https://github.com/user-attachments/assets/786c53f8-5e3d-475a-97ce-ab1a89d661d0" />

<img width="2366" height="1468" alt="rq1_a_path_moderation" src="https://github.com/user-attachments/assets/7e6581b3-0a85-4177-b511-92feb3d45662" />

<img width="1760" height="1100" alt="rq2_diminishing_returns_curve" src="https://github.com/user-attachments/assets/85320757-078a-45e2-a28e-4e51bc624804" />

<img width="1320" height="880" alt="rq3_high_visibility_effect" src="https://github.com/user-attachments/assets/f747c18f-cffb-4f78-a9da-22a2ed77280c" />


---

## Repository structure

```text
social-proof-to-action/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── AUTHORS.md
├── .gitignore
├── requirements.txt
├── PROJECT_SUMMARY.md
├── REPRODUCIBILITY.md
├── MANIFEST.md
│
├── assets/
│   ├── cover/
│   │   └── project_cover.png
│   ├── figures/
│   │   ├── conceptual_model.png
│   │   ├── rq1_authority_effect.png
│   │   ├── rq2_diminishing_returns.png
│   │   ├── rq3_high_visibility_effect.png
│   │   └── slide_preview.png
│   └── branding/
│       └── csun_banner.png
│
├── data/
│   ├── README.md
│   ├── codebook/
│   │   ├── dataset_codebook.csv
│   │   ├── dataset_profile.md
│   │   └── variable_dictionary.md
│   ├── raw/
│   │   └── dataset_algorithmic_persuasion_10000.xlsx
│   └── processed/
│       └── dataset_feature_engineered.csv
│
├── docs/
│   ├── paper/
│   │   ├── research_paper_final.docx
│   │   ├── research_paper_final.pdf
│   │  ── research_paper_final.pdf
│ └── abstract.pdf
│   ├── presentation/
│   │   ├── CSUN_BANA698_Final_Presentation_Social_Proof_to_Action.pptx
│   │   ├── CSUN_BANA698_Final_Presentation_Social_Proof_to_Action.pdf
│   │   └── presentation_script.md
│   ├── methodology/
│   │   ├── research_design.md
│   │   ├── measures_and_operationalization.md
│   │   └── model_logic.md
│   ├── summaries/
│   │   ├── project_summary.md
│   │   ├── paper_summary.md
│   │   └── references.md
│   └── appendix/
│       └── appendix_materials.pdf
│
├── outputs/
│   ├── README.md
│   ├── descriptive/
│   │   ├── descriptive_statistics_output.xlsx
│   │   ├── descriptive_summary.txt
│   │   ├── outliers_report.xlsx
│   │   └── figures/
│   │       ├── log_reach_hist.png
│   │       ├── reach_hist.png
│   │       ├── persuasive_power_index_hist.png
│   │       └── authority_log_hist.png
│   ├── rq1/
│   │   ├── rq1_a_path.csv
│   │   ├── rq1_b_path.csv
│   │   ├── rq1_conditional_indirect_effects.csv
│   │   ├── rq1_summary.txt
│   │   └── figures/
│   │       ├── rq1_a_path_moderation.png
│   │       └── rq1_b_path_moderation.png
│   ├── rq2/
│   │   ├── rq2_diminishing_returns.csv
│   │   ├── rq2_summary.txt
│   │   └── figures/
│   │       └── rq2_diminishing_returns_curve.png
│   ├── rq3/
│   │   ├── rq3_high_visibility.csv
│   │   ├── rq3_summary.txt
│   │   └── figures/
│   │       └── rq3_high_visibility_effect.png
│   └── summary/
│       ├── hypothesis_summary_all_rqs.csv
│       ├── summary_of_hypothesis_testing_results.csv
│       └── executive_findings.md
│
├── src/
│   ├── README.md
│   ├── feature_engineering.py
│   ├── descriptive_statistics.py
│   ├── rq1_mediation_moderation.py
│   ├── hypothesis_tests_rq1_rq2_rq3.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io_helpers.py
│   │   └── plotting_helpers.py
│   └── legacy/
│       ├── main_analysis.py
│       ├── visualization.py
│       ├── rq1_visualizations.py
│       ├── outliers.py
│       ├── amplification_translate.py
│       └── summary_hypothesis_testing_results.py
│
├── scripts/
│   ├── run_pipeline.sh
│   ├── generate_outputs.sh
│   └── setup_project.sh
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── results_validation.ipynb
│
└── archive/
    ├── README.md
    ├── drafts/
    ├── administrative/
    └── reference_materials/
```
## pipe line
flowchart TD

    A[Research Problem<br/>Why does visibility not always become meaningful engagement?] --> B[Theory Lens<br/>HSM + Source Credibility Theory]
    B --> C[Research Questions<br/>RQ1: Mechanism + Authority<br/>RQ2: Diminishing Returns<br/>RQ3: High-Visibility Condition]

    C --> D[Dataset Intake<br/>10,000 posts<br/>800 influencers<br/>Timestamped post-level data]
    D --> E[Data Understanding<br/>Review codebook, variables, structure, and missingness]
    E --> F[Data Preparation<br/>Cleaning, transformations, factor encoding, casewise exclusion]
    F --> G[Feature Engineering<br/>Persuasive power index<br/>log reach<br/>log reach squared<br/>authority_log<br/>very_high_visibility]

    G --> H[Descriptive Analysis<br/>Summary statistics<br/>Distribution checks<br/>Initial diagnostics]

    H --> I[RQ1 Analysis<br/>Mediation + Moderation Models]
    I --> I1[a-path<br/>Persuasive power → Amplification]
    I --> I2[b-path<br/>Amplification → High-effort engagement]
    I --> I3[Conditional indirect effects<br/>by authority level]

    H --> J[RQ2 Analysis<br/>Quadratic Reach Model]
    J --> J1[Estimate linear + squared reach terms]
    J --> J2[Calculate turning point]
    J --> J3[Test diminishing marginal returns]

    H --> K[RQ3 Analysis<br/>High-Visibility Interaction Model]
    K --> K1[Baseline engagement model]
    K --> K2[Persuasive power × visibility interaction]
    K --> K3[Test whether persuasive power weakens under extreme exposure]

    I3 --> L[Robustness Checks]
    J3 --> L
    K3 --> L
    L --> L1[Alternative model checks]
    L --> L2[Low/high reach segment checks]
    L --> L3[Engagement-efficiency model]
    L --> L4[Diagnostics: multicollinearity, heteroskedasticity, residual review]

    L --> M[Results Synthesis]
    M --> M1[Authority strengthens the pathway]
    M --> M2[Reach shows diminishing returns]
    M --> M3[Very high visibility weakens persuasive differentiation]

    M --> N[Final Deliverables]
    N --> N1[Research Paper]
    N --> N2[Presentation Slides]
    N --> N3[Figures and Tables]
    N --> N4[GitHub Repository]
<img width="4941" height="3958" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/d1b3eeda-e962-4079-ba15-af0a20e046c7" />

-----

### Folder guide

- **assets/** — cover image, conceptual model, and selected figures for GitHub display
- **data/** — raw workbook, processed file, and extracted codebook
- **docs/** — presentation deck, presentation PDF, abstracts, and repository summaries
- **outputs/** — descriptive outputs and RQ1–RQ3 result files
- **src/** — main scripts plus preserved legacy code
- **scripts/** — helper script to run the main pipeline
- **archive/** — drafts, reference materials, and administrative artifacts from the original project bundle

---

## How to use this repository

### Quick access

- **Presentation (PPTX):** `docs/presentations/CSUN_BANA698_Final_Presentation_Social_Proof_to_Action.pptx`
- **Presentation (PDF):** `docs/presentations/CSUN_BANA698_Final_Presentation_Social_Proof_to_Action.pdf`
- **Project summary:** `PROJECT_SUMMARY.md`
- **Dataset profile:** `docs/summaries/dataset_profile.md`
- **Variable dictionary:** `docs/summaries/variable_dictionary.md`

### Reproduce the main analysis

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the main workflow:

```bash
bash scripts/run_pipeline.sh
```

Or run the scripts manually:

```bash
python src/feature_engineering.py
python src/descriptive_statistics.py
python src/rq1_mediation_moderation.py
python src/hypothesis_tests_rq1_rq2_rq3.py
```

---

## Important sharing notes

- The project is **observational**, so results should be interpreted as **associative rather than causal**.
- The repository includes a raw instructional dataset and archived course/project materials. **Review permissions before making the repository public.**
- MacOS metadata files and the embedded `.venv` from the source bundle were removed from this GitHub-ready version for cleanliness.

---

## Citation

See `CITATION.cff` for repository citation metadata.

---

## Authors

See `AUTHORS.md`.

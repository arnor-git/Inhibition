```markdown
# ReStroop: Eye-Tracking Analysis Framework for Inhibitory Control Evaluation

This repository contains the official algorithmic pipeline and data processing framework for ReStroop, an eye-tracking-driven educational game and cognitive assessment system. ReStroop presents a recycling-themed Stroop paradigm designed to evaluate and train inhibitory control skills in children.

The framework processes raw, high-frequency gaze time-series streams through unified fixation classification, Area of Interest (AOI) spatial mapping, performance modelling, and automated multi-dimensional risk assessment.

---

## 🌟 Key Features

* Dual-Threshold Eye Movement Detection: Combines Velocity-Threshold Identification (I-VT) and spatial density clustering (DBSCAN/K-Means) to reliably filter gaze noise and extract fixations, saccades, and visual search paths.
* Dynamic AOI Analytics: Maps spatiotemporal gaze fixations onto dynamic game elements (target bins, distractor items, feedback banners) in real time.
* Multi-Dimensional Risk Engine:** Calculates real-time risk scores across task relevance, attention scatter, and processing efficiency metrics to detect cognitive overload vs. adaptive learning.
* Personalised Intervention Generator: Automatically triggers evidence-based recommendations, including attention regulation prompts, adaptive difficulty tuning, and environmental modifications for teachers and psychologists.
* Unified Telemetry Logger: Integrates seamlessly with Unity 6 game engines and Tobii Pro hardware environments.



## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Tobii Eye Tracking SDK / Tobii Pro SDK


## 🚀 Usage

### 1. Processing Raw Gaze Data

Run the pipeline to process raw gaze logs into classified fixations, AOI heatmaps, and risk metrics:

```python
from restroop_analytics import GazePipeline, RiskEvaluator

# Initialise pipeline with I-VT velocity threshold
pipeline = GazePipeline(velocity_threshold=30.0, min_fixation_duration=100)

# Load and process raw CSV eye-tracking log
gaze_data = pipeline.load_raw_log("data/raw/participant_01_level2.csv")
fixations = pipeline.extract_fixations(gaze_data)

# Run Multi-Dimensional Risk Engine
evaluator = RiskEvaluator(aoi_config="config/restroop_aois.json")
risk_report = evaluator.compute_risk_profile(fixations)

print(f"Attention Scatter Index: {risk_report.scatter_score}")
print(f"Recommended Intervention: {risk_report.intervention_plan}")

```

### 2. Generating Clinical & Educational Reports

Generate exportable PDF/JSON analytical summaries for teachers and domain specialists:

```bash
python scripts/generate_report.py --input data/processed/participant_01.json --output reports/P01_Assessment.pdf

```

---

## 📊 Analytical Metrics Overview

| Metric Category | Indicators Measured | Target Evaluation |
| --- | --- | --- |
| **Task Relevance** | Fixation Proportion on Target vs. Distractors | Identifies visual distraction susceptibility |
| **Attention Scatter** | Spatial Entropy, Saccadic Amplitude Variance | Measures visual search instability |
| **Processing Efficiency** | Time-to-First-Fixation (TTFF), Fixation Duration | Assesses cognitive load & decision-speed |
| **Risk Status** | High / Moderate / Low Risk Categorization | Triggers real-time difficulty adjustment |

---

## 📄 Associated Publications & Citations

If you use this codebase or methodology in your research, please cite the following publication:

```bibtex
@article{rehman2025personalized,
  title={Personalized Inhibition Training with Eye-Tracking: Enhancing Student Learning and Teacher Assessment in Educational Games},
  author={Rehman, Abdul and Heldal, Ilona and Stilwell, David and Ferreira, Paulo C. and Lin, Jerry Chun-Wei},
  journal={arXiv preprint arXiv:2509.08357},
  year={2025}
}

```

### Related Artifacts

* **`VisiTrail`**: Visual analytics dashboard for time-series gaze telemetry (*Scientific Reports*, 2026).

---

*License: This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.*

```

```

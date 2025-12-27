# Network Intrusion Detection using QML

> This project allows for a comparitive study of Network Intrusion Detection (NID) using both classical and quantum machine learning.

## Project Overview

This project can be treated as a survey project that helps to analyse the performance of the state of the art classication models in the classical ML universe against their counterparts in the Quantum world.

This will also take into consideration top of the line classication models from the Quantum world (whose counterparts may or may not exist in the classical world).

---

## Goals & Success Metrics

**Primary goal**

* Evaluate whether quantum ML methods can improve intrusion detection over tuned classical baselines.

**Research Questions**

* Which Quantum based method outperform classical methods (RF/DT/SVM) on reduced features?​

* What conditions (small-data, specific attack classes, noisy hardware) favour QML?​

* Which pre-processing choices prevent attack signal recognition in QML?

**Metrics**

*  F1-score / Accuracy / Precision / Recall on a test set

---

## Version & Requirements
| Component | Version |
|------------|----------|
| **Python** | 3.12 |
| **qiskit** | 1.2.4 |

---

The other components can be versioned as is compatible with the above python and qiskit versions.

## Repository Structure

```
README.md
src/
  data/               # data being used
  models/             # model architectures / training
  plot/               # Plots of the results
```

---

## Setup & Quickstart

1. Clone the repo:

```bash
```
---

## License

This project is licensed under the terms of the MIT license.

---

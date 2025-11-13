# Chapter 4 – Shor’s Algorithm and Post-Quantum Order Finding (Public Demos)

This directory contains two **fully classical**, **public** companion demos for
Chapter 4 of *Post-Quantum Superintelligence*. These scripts illustrate the
core mathematical ideas behind Shor-style order finding, without revealing the
post-quantum operator machinery used in the private PQSI core.

## 1. Structured Order Finding (Classical DFT)

**File:** `shor_structured_dft_demo.py`

This script illustrates the structured case where modular multiplication
`a^k mod N` can be computed directly. The sequence is embedded into a complex
signal, a discrete Fourier transform (DFT) is applied, and the dominant
frequency reveals the candidate order.

This mirrors the geometry of Shor’s quantum Fourier transform (QFT), but
entirely on classical hardware.

Run it:

```bash
python3 shor_structured_dft_demo.py
python3 shor_structured_dft_demo.py --N 21 --a 2 --K 64

## 2. Unstructured Correlation Operator Demo

**File:** `shor_unstructured_correlation_demo.py'

This is a tiny unstructured toy example. Given a signal with a hidden period,
the script constructs a correlation-style operator from cyclic shifts and
computes its leading eigenvector. A DFT of that eigenvector exposes the
dominant frequency, revealing the underlying period.

Run it:

```bash
python3 shor_unstructured_correlation_demo.py


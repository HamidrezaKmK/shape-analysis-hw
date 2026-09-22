# Shape Analysis (6.8410) homework

Homework code for MIT **6.8410 Shape Analysis**, Fall 2026 ([course page](https://groups.csail.mit.edu/gdpgroup/68410_fall_2026.html)).

## Goal

The course hands out MATLAB starter code. This repository keeps a **Python translation of every starter file next to the original**, so the assignments can be solved in numpy/scipy/matplotlib while staying line-for-line comparable with what the course distributes. Each `hwN/` folder holds:

- the handout (`hwN.pdf`),
- the original MATLAB starter files, untouched,
- Python translations with the same structure and the same `### YOUR CODE HERE ###` placeholders, so problems map one-to-one,
- (later) the worked solutions and the figures produced for the write-up.

Write-ups, derivations and lecture notes live in the companion Lockedin research bubble (Mana); this repository is the code side of that project.

## Layout

```
hw1/                Discrete and smooth curves (variational calculus, discrete curvature, discrete elastic rods)
  hw1.pdf           handout
  discreteCurve.m   Problem 2 starter (MATLAB)
  elasticRods.m     Problem 3 starter (MATLAB)
  discrete_curve.py Problem 2 starter, Python
  elastic_rods.py   Problem 3 starter, Python
  requirements.txt
```

## Running

```bash
cd hw1
python -m venv .venv && source .venv/bin/activate   # or: uv venv .venv && source .venv/bin/activate
pip install -r requirements.txt                     # or: uv pip install -r requirements.txt
python discrete_curve.py
python elastic_rods.py [bendModulus] [twistModulus] [totalTwist]
```

Both scripts open matplotlib windows; set `MPLBACKEND=Agg` to run them headless.

## Conventions kept from the MATLAB code

- Vertex arrays are `(dim, n)`, one column per vertex (MATLAB layout), so indices match the handout and the paper.
- `elastic_rods.py`: edge `i` runs from vertex `i` to `i+1` (mod n); `parallelTransport[:, :, i]` is the rotation $P_i$ taking the Bishop frame of edge `i-1` onto edge `i`; `bishopFrame` is `(3, 2, n+1)` with the last entry the frame transported all the way around (holonomy). MATLAB's `axang2rotm` is replaced by Rodrigues' formula.

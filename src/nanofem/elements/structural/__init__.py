"""nanofem.elements.structural.

Closed-form structural elements behind the common Element interface (ADR-002).

Responsibilities
----------------
- Bar, truss, Euler-Bernoulli and Timoshenko beams, frame; local frames and transformations
  (E-10)

Future modules
--------------
- bar.py (implemented: Bar - closed-form axial element, 1-D-in-1-D embedding only)
- truss.py
- beam_eb.py (implemented: EulerBernoulliBeam - closed-form bending element, 1-D-in-1-D
  embedding only)
- beam_timoshenko.py (implemented: TimoshenkoBeam - closed-form selective-reduced-integration
  bending+shear element, 1-D-in-1-D embedding only)
- frame.py

TODO
----
- TODO(phase-1): Truss2D/Frame2D (direction-cosine transformation) and the remaining
  structural suite
"""

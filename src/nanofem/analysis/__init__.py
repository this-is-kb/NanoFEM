"""nanofem.analysis.

Orchestration only; physics-free by ADR-008 (SDS 2.18).

Responsibilities
----------------
- Template method: validate -> number -> assemble by role -> constrain -> solve -> package
- Static, modal, buckling, transient; optimization slot; immutable results; run report (SDS
  Section 13)

Future modules
--------------
- base.py (implemented: AnalysisBase)
- static.py (implemented: LinearStaticAnalysis, StaticResult - real since the walking skeleton)
- modal.py, buckling.py, transient.py (metadata/options only; run() still refuses per phase-1 scope)
- results.py (ModalResult/BucklingResult/TransientResult placeholders only)
- optimization/

TODO
----
- TODO(phase-3): ModalAnalysis/LinearBucklingAnalysis/TransientAnalysis real run() bodies
"""

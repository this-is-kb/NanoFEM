"""nanofem.core.

Model facade, DOF bookkeeping, field definitions, and the class registry.

Responsibilities
----------------
- Model facade holding mesh, element sets, materials, constraints, load cases
- DofHandler: (node, field, component) -> global equation number (SDS C-2, C-5)
- FieldSpec declarations; Registry for plugin/factory lookup (SDS Section 12)

Future modules
--------------
- model.py (implemented: Model, DomainDefinition; validate() walk real)
- dof_handler.py (implemented: DofHandler, Dof; deterministic SDS C-5 numbering + fingerprint())
- fields.py (implemented: FieldSpec, factories)
- registry.py (Registry: declared, no current consumer - SDS Section 12 plugin lookup awaits a
  first plugin)

TODO
----
- TODO(phase-2+): a first real Registry consumer (element/theory/solver plugin lookup)
"""

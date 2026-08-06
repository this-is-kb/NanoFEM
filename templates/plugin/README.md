# NanoFEM plugin template

Copy this directory to start a third-party extension that registers with
NanoFEM through entry points — no fork, no changes to NanoFEM source
(SDS Section 12).

Entry-point groups: `nanofem.elements`, `nanofem.theories`,
`nanofem.constitutive`, `nanofem.kernels`, `nanofem.materials`,
`nanofem.solvers`, `nanofem.analyses`.

Rules: keys MUST be namespaced (`"yourgroup.thing"`); duplicate keys are an
error; your plugin is valid iff it satisfies the corresponding SDS contract
section. The conformance kit (phase 0.5+) makes "certified against
nanofem-sds vX.Y" a machine-checkable claim — run it in your own CI.

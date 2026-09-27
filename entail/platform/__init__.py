"""The platform (entail 2.0, LIBRARY_DESIGN.md 13; ROADMAP product track): a project's runs shown as the nodes of its
workflow, from the record files the core writes. It reads records only - it never runs inside an engine - so it adds
nothing to an engine's cost (principle 10), and it needs nothing outside the Python standard library.

  graph   the node model: record lines -> launches -> nodes with their state, progress and decisions (P1)
"""

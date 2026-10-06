"""Shared, model-independent groundwater risk category labels."""

LABELS = ["Low", "Moderate", "High", "Very High"]
LABEL_TO_INT = {label: index for index, label in enumerate(LABELS)}

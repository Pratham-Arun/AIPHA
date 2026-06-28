"""
Medical Calculator Tool  –  Phase 7.1
───────────────────────────────────────
Deterministic healthcare mathematics — no LLM required for the calculations.

Supported functions:
  1. BMI Calculator          → BMI value + WHO category
  2. Basal Metabolic Rate    → Daily calorie needs via Mifflin–St Jeor equation
  3. Water Intake            → Recommended daily hydration
  4. Ideal Body Weight       → Devine formula
  5. Body Surface Area       → Mosteller formula (clinical dosing reference)

Used by:
  - Nutrition Agent  (BMI, BMR, water intake, ideal weight)

All calculations are deterministic and never require the LLM.
The LLM is only used *after* to explain the result in plain language.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class BMIResult:
    bmi: float
    category: str
    height_cm: float
    weight_kg: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bmi": round(self.bmi, 1),
            "category": self.category,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
        }

    def __str__(self) -> str:
        return (
            f"BMI: {self.bmi:.1f}  |  Category: {self.category}\n"
            f"Height: {self.height_cm} cm  |  Weight: {self.weight_kg} kg"
        )


@dataclass
class BMRResult:
    bmr_calories: float
    gender: str
    age: int
    height_cm: float
    weight_kg: float
    activity_calories: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bmr_calories": round(self.bmr_calories, 0),
            "gender": self.gender,
            "age": self.age,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "activity_calories": {k: round(v, 0) for k, v in self.activity_calories.items()},
        }

    def __str__(self) -> str:
        lines = [
            f"Basal Metabolic Rate (BMR): {self.bmr_calories:.0f} kcal/day",
            f"Gender: {self.gender}  |  Age: {self.age}  |  "
            f"Height: {self.height_cm} cm  |  Weight: {self.weight_kg} kg",
            "",
            "Daily Calorie Needs by Activity Level:",
        ]
        labels = {
            "sedentary": "Sedentary (desk job, little exercise)",
            "light":     "Lightly active (1–3 days/week exercise)",
            "moderate":  "Moderately active (3–5 days/week)",
            "active":    "Very active (6–7 days/week hard exercise)",
            "extra":     "Extra active (physical job or 2x training)",
        }
        for key, cal in self.activity_calories.items():
            lines.append(f"  {labels.get(key, key)}: {cal:.0f} kcal/day")
        return "\n".join(lines)


@dataclass
class WaterIntakeResult:
    recommended_litres: float
    weight_kg: float
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_litres": round(self.recommended_litres, 2),
            "recommended_ml": round(self.recommended_litres * 1000, 0),
            "weight_kg": self.weight_kg,
            "note": self.note,
        }

    def __str__(self) -> str:
        return (
            f"Recommended Water Intake: {self.recommended_litres:.2f} L/day "
            f"({self.recommended_litres * 1000:.0f} mL/day)\n"
            f"Based on weight: {self.weight_kg} kg\n"
            f"{self.note}"
        )


@dataclass
class IdealWeightResult:
    ideal_weight_kg: float
    gender: str
    height_cm: float
    formula: str = "Devine"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ideal_weight_kg": round(self.ideal_weight_kg, 1),
            "gender": self.gender,
            "height_cm": self.height_cm,
            "formula": self.formula,
        }

    def __str__(self) -> str:
        return (
            f"Ideal Body Weight ({self.formula} formula): {self.ideal_weight_kg:.1f} kg\n"
            f"Gender: {self.gender}  |  Height: {self.height_cm} cm"
        )


@dataclass
class BSAResult:
    bsa_m2: float
    height_cm: float
    weight_kg: float
    formula: str = "Mosteller"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bsa_m2": round(self.bsa_m2, 3),
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "formula": self.formula,
        }

    def __str__(self) -> str:
        return (
            f"Body Surface Area ({self.formula}): {self.bsa_m2:.3f} m²\n"
            f"Height: {self.height_cm} cm  |  Weight: {self.weight_kg} kg"
        )


# ── BMI category thresholds (WHO) ─────────────────────────────────────────────
_BMI_CATEGORIES = [
    (0.0,  16.0,  "Severe Thinness"),
    (16.0, 17.0,  "Moderate Thinness"),
    (17.0, 18.5,  "Mild Thinness / Underweight"),
    (18.5, 25.0,  "Normal / Healthy Weight"),
    (25.0, 30.0,  "Overweight"),
    (30.0, 35.0,  "Obese Class I"),
    (35.0, 40.0,  "Obese Class II"),
    (40.0, float("inf"), "Obese Class III (Morbidly Obese)"),
]

# Activity multipliers for TDEE from BMR
_ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "light":     1.375,
    "moderate":  1.55,
    "active":    1.725,
    "extra":     1.9,
}


class MedicalCalculatorTool:
    """
    Performs deterministic healthcare calculations.

    All methods validate their inputs and raise ValueError with clear messages
    on invalid data, so the caller can surface a useful error without an LLM call.

    Usage:
        calc = MedicalCalculatorTool()
        result = calc.bmi(height_cm=181, weight_kg=52)
        print(result)   # BMI: 15.9  |  Category: Mild Thinness / Underweight
    """

    # ── BMI ───────────────────────────────────────────────────────────────────

    def bmi(self, height_cm: float, weight_kg: float) -> BMIResult:
        """
        Calculate Body Mass Index using the standard WHO formula.

        Args:
            height_cm: Height in centimetres (must be > 0).
            weight_kg: Weight in kilograms (must be > 0).

        Returns:
            BMIResult with value and WHO category.
        """
        self._validate_positive(height_cm, "height_cm")
        self._validate_positive(weight_kg, "weight_kg")

        height_m = height_cm / 100.0
        bmi_value = weight_kg / (height_m ** 2)
        category = self._bmi_category(bmi_value)
        return BMIResult(bmi=bmi_value, category=category,
                         height_cm=height_cm, weight_kg=weight_kg)

    # ── BMR (Mifflin–St Jeor) ─────────────────────────────────────────────────

    def bmr(
        self,
        age: int,
        height_cm: float,
        weight_kg: float,
        gender: str,
    ) -> BMRResult:
        """
        Calculate Basal Metabolic Rate using the Mifflin–St Jeor equation.

        Args:
            age:       Age in years (1–120).
            height_cm: Height in centimetres.
            weight_kg: Weight in kilograms.
            gender:    "male" or "female" (case-insensitive).

        Returns:
            BMRResult with base calories and TDEE estimates by activity level.
        """
        self._validate_positive(age, "age")
        self._validate_positive(height_cm, "height_cm")
        self._validate_positive(weight_kg, "weight_kg")

        gender_lower = gender.lower().strip()
        if gender_lower not in ("male", "female", "m", "f"):
            raise ValueError(
                f"Invalid gender '{gender}'. Use 'male' or 'female'."
            )
        is_male = gender_lower in ("male", "m")

        # Mifflin–St Jeor
        bmr_val = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
        bmr_val += 5 if is_male else -161

        activity_calories = {
            level: round(bmr_val * mult, 1)
            for level, mult in _ACTIVITY_MULTIPLIERS.items()
        }

        return BMRResult(
            bmr_calories=round(bmr_val, 1),
            gender="Male" if is_male else "Female",
            age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            activity_calories=activity_calories,
        )

    # ── Water Intake ──────────────────────────────────────────────────────────

    def water_intake(self, weight_kg: float) -> WaterIntakeResult:
        """
        Estimate recommended daily water intake (35 mL per kg body weight).

        Args:
            weight_kg: Body weight in kilograms.

        Returns:
            WaterIntakeResult in litres and millilitres.
        """
        self._validate_positive(weight_kg, "weight_kg")
        litres = (weight_kg * 35) / 1000.0
        return WaterIntakeResult(
            recommended_litres=litres,
            weight_kg=weight_kg,
            note="Based on the standard 35 mL/kg guideline. "
                 "Actual needs vary with activity, climate, and health status.",
        )

    # ── Ideal Body Weight (Devine formula) ───────────────────────────────────

    def ideal_body_weight(self, height_cm: float, gender: str) -> IdealWeightResult:
        """
        Calculate Ideal Body Weight using the Devine formula.

        Args:
            height_cm: Height in centimetres.
            gender:    "male" or "female" (case-insensitive).

        Returns:
            IdealWeightResult in kilograms.
        """
        self._validate_positive(height_cm, "height_cm")
        gender_lower = gender.lower().strip()
        if gender_lower not in ("male", "female", "m", "f"):
            raise ValueError(f"Invalid gender '{gender}'. Use 'male' or 'female'.")
        is_male = gender_lower in ("male", "m")

        height_in = height_cm / 2.54
        inches_over_5ft = max(0, height_in - 60)

        if is_male:
            ibw = 50.0 + 2.3 * inches_over_5ft
        else:
            ibw = 45.5 + 2.3 * inches_over_5ft

        return IdealWeightResult(
            ideal_weight_kg=round(ibw, 1),
            gender="Male" if is_male else "Female",
            height_cm=height_cm,
        )

    # ── Body Surface Area (Mosteller) ────────────────────────────────────────

    def body_surface_area(self, height_cm: float, weight_kg: float) -> BSAResult:
        """
        Calculate Body Surface Area using the Mosteller formula.
        Commonly used for clinical drug dosing.

        BSA (m²) = sqrt((height_cm × weight_kg) / 3600)

        Args:
            height_cm: Height in centimetres.
            weight_kg: Weight in kilograms.

        Returns:
            BSAResult in square metres.
        """
        self._validate_positive(height_cm, "height_cm")
        self._validate_positive(weight_kg, "weight_kg")
        bsa = math.sqrt((height_cm * weight_kg) / 3600.0)
        return BSAResult(bsa_m2=round(bsa, 3), height_cm=height_cm, weight_kg=weight_kg)

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def run(self, calculation: str, **kwargs):
        """
        Unified entry point.  Dispatches to the correct calculation method
        based on *calculation* name.

        Supported values:
          "bmi"               → bmi(height_cm, weight_kg)
          "bmr"               → bmr(age, height_cm, weight_kg, gender)
          "water_intake"      → water_intake(weight_kg)
          "ideal_weight"      → ideal_body_weight(height_cm, gender)
          "bsa"               → body_surface_area(height_cm, weight_kg)

        Returns:
            The appropriate result dataclass instance.

        Raises:
            ValueError: On unknown calculation name or missing / invalid params.
        """
        calculation = calculation.lower().strip().replace(" ", "_")
        dispatch = {
            "bmi":          self.bmi,
            "bmr":          self.bmr,
            "water_intake": self.water_intake,
            "ideal_weight": self.ideal_body_weight,
            "ibw":          self.ideal_body_weight,
            "bsa":          self.body_surface_area,
        }
        if calculation not in dispatch:
            supported = ", ".join(dispatch.keys())
            raise ValueError(
                f"Unknown calculation '{calculation}'. "
                f"Supported: {supported}"
            )
        return dispatch[calculation](**kwargs)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_positive(value, name: str) -> None:
        if value is None:
            raise ValueError(f"'{name}' is required.")
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number, got: {value!r}")
        if v <= 0:
            raise ValueError(f"'{name}' must be greater than zero, got: {v}")

    @staticmethod
    def _bmi_category(bmi: float) -> str:
        for lower, upper, label in _BMI_CATEGORIES:
            if lower <= bmi < upper:
                return label
        return "Unknown"

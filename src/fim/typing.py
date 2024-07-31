from pydantic import BaseModel, ValidationInfo, field_validator


class FunctionExpressionParameters(BaseModel):
    operators: str
    max_ops: int
    max_int: int
    max_len: int
    int_base: int
    balanced: bool
    precision: int
    positive: bool
    rewrite_functions: str
    leaf_probs: list[int]
    n_variables: int
    n_coefficients: int

    @field_validator("n_variables")
    def val_n_variables(cls, v: int, info: ValidationInfo):
        if v < 1 and v > 3:
            raise ValueError("n_variables must be between 1 and 3")
        return v

    @field_validator("n_coefficients")
    def val_n_coefficients(cls, v: int, info: ValidationInfo):
        if v < 0 and v > 10:
            raise ValueError("n_coefficients must be between 0 and 10")
        return v

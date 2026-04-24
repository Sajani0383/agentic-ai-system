import warnings


# Keep terminal output focused on actionable project issues.
warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*Python version 3\.9 past its end of life.*",
)
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message=r".*unclosed event loop.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*@model_validator.*mode='after'.*",
)

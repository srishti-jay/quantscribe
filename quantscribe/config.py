"""
QuantScribe global configuration.

All paths, model names, and tunable parameters live here.
Import this module instead of hardcoding values anywhere.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# ── Project Root ──
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # ── API Keys ──
    google_api_key: str = Field(default="", description="Gemini API key from aistudio.google.com")

    # ── Paths ──
    pdf_input_dir: Path = Field(default=PROJECT_ROOT / "data" / "pdfs")
    index_dir: Path = Field(default=PROJECT_ROOT / "indices" / "active")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs")
    gold_standard_dir: Path = Field(default=PROJECT_ROOT / "eval" / "gold_standard")

    # ── Embedding Model ──
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)
    embedding_max_tokens: int = Field(default=256)
    embedding_batch_size: int = Field(default=64)

    # ── LLM ──
    llm_model: str = Field(default="gemini-2.5-flash")
    llm_temperature: float = Field(default=0.0)
    llm_max_retries: int = Field(default=3)
    llm_max_output_tokens: int = Field(default=4096)

    # ── Chunking ──
    narrative_chunk_size_words: int = Field(default=300)
    narrative_overlap_words: int = Field(default=100)
    narrative_max_chunk_words: int = Field(default=500)
    table_max_tokens: int = Field(default=1024)

    # ── Retrieval ──
    top_k_per_bank: int = Field(default=5)

    # ── ETL ──
    parse_version: str = Field(default="etl_v1.0.0")

    # ── Page Classification Thresholds ──
    # Tune these empirically on 10 sample pages per bank
    graphical_image_threshold: int = Field(default=2)
    graphical_text_block_threshold: int = Field(default=3)
    mixed_text_block_threshold: int = Field(default=5)

    # ── Evaluation ──
    numerical_tolerance: float = Field(
        default=0.005,
        description="Relative tolerance for numerical accuracy (0.5%)"
    )

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# ── Singleton ──
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# ── Supported Banks ──
SUPPORTED_BANKS: list[str] = [
    "HDFC_BANK",
    "SBI",
    "ICICI_BANK",
    "AXIS_BANK",
]

# ── Known Section Headers ──
KNOWN_SECTIONS: list[str] = [
    "Management Discussion and Analysis",
    "Management Discussion & Analysis",
    "MD&A",
    "Risk Management",
    "Credit Risk",
    "Market Risk",
    "Operational Risk",
    "Liquidity Risk",
    "Capital Adequacy",
    "Basel III Disclosures",
    "Basel III Pillar 3 Disclosures",
    "Asset Quality",
    "Corporate Governance",
    "Auditor's Report",
    "Independent Auditor's Report",
    "Notes to Financial Statements",
    "Profit and Loss",
    "Balance Sheet",
    "Cash Flow Statement",
    "Schedules to Financial Statements",
    "Directors' Report",
    "Report on Corporate Governance",
    "Segment Reporting",
    "Related Party Transactions",
]

# ── Macro Theme Taxonomy ──
THEME_TAXONOMY: dict[str, dict] = {
    "credit_risk": {
        "name": "Credit Risk Exposure",
        "target_metrics": [
            "gross_npa_ratio", "net_npa_ratio",
            "provision_coverage_ratio", "slippage_ratio",
        ],
        "source_sections": ["MD&A", "Asset Quality", "Risk Management"],
        "query_anchor": "credit risk gross NPA net NPA provision coverage slippage ratio asset quality",
    },
    "liquidity_risk": {
        "name": "Liquidity Risk",
        "target_metrics": [
            "lcr_percent", "nsfr_percent", "loan_to_deposit_ratio",
        ],
        "source_sections": ["Risk Management", "Basel III Disclosures"],
        "query_anchor": "liquidity risk LCR NSFR loan to deposit ratio funding stability",
    },
    "unsecured_lending": {
        "name": "Unsecured Lending Exposure",
        "target_metrics": [
            "unsecured_loan_percent", "personal_loan_growth",
            "credit_card_npa",
        ],
        "source_sections": ["MD&A", "Segment Reporting"],
        "query_anchor": "unsecured lending personal loan credit card exposure retail loan growth",
    },
    "capital_adequacy": {
        "name": "Capital Adequacy",
        "target_metrics": [
            "cet1_ratio", "tier1_ratio", "total_car", "rwa_growth",
        ],
        "source_sections": ["Basel III Disclosures", "Capital Adequacy"],
        "query_anchor": "capital adequacy CET1 tier 1 CAR CRAR risk weighted assets Basel III",
    },
    "market_risk": {
        "name": "Market Risk",
        "target_metrics": [
            "var_10day", "duration_gap", "trading_book_size", "mtm_losses",
        ],
        "source_sections": ["Risk Management", "Market Risk"],
        "query_anchor": "market risk value at risk VaR duration gap trading book MTM losses interest rate risk",
    },
    "operational_risk": {
        "name": "Operational Risk",
        "target_metrics": [
            "oprisk_rwa", "fraud_losses", "cyber_incidents", "bcp_status",
        ],
        "source_sections": ["Risk Management", "Operational Risk"],
        "query_anchor": "operational risk fraud losses cyber incidents business continuity RWA",
    },
    "asset_quality_trend": {
        "name": "Asset Quality Trends",
        "target_metrics": [
            "npa_opening", "npa_additions", "npa_recoveries",
            "npa_closing", "writeoff_rate",
        ],
        "source_sections": ["Schedules to Financial Statements", "Asset Quality"],
        "query_anchor": "asset quality NPA movement additions recoveries write-off upgrading restructured",
    },
}

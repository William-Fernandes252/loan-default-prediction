"""Tests for the data transformers module."""

import polars as pl
import pytest

from experiments.core.data import Dataset
from experiments.core.data.base import BaseDataTransformer
from experiments.core.data.corporate_credit import CorporateCreditTransformer
from experiments.core.data.lending_club import LendingClubTransformer
from experiments.core.data.protocols import DataTransformer
from experiments.core.data.taiwan_credit import TaiwanCreditTransformer


class DescribeBaseDataTransformer:
    """Tests for BaseDataTransformer abstract class."""

    def it_requires_subclass_to_implement_dataset_name(self) -> None:
        """Verify abstract property enforcement."""
        with pytest.raises(TypeError, match="abstract"):

            class IncompleteTransformer(BaseDataTransformer):
                def _apply_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
                    return df

            IncompleteTransformer()  # type: ignore[abstract]

    def it_requires_subclass_to_implement_apply_transformations(self) -> None:
        """Verify abstract method enforcement."""
        with pytest.raises(TypeError, match="abstract"):

            class IncompleteTransformer(BaseDataTransformer):
                @property
                def dataset_name(self) -> str:
                    return "test"

            IncompleteTransformer()  # type: ignore[abstract]

    def it_accepts_use_gpu_parameter(self) -> None:
        """Verify GPU flag is stored correctly."""

        class ConcreteTransformer(BaseDataTransformer):
            @property
            def dataset_name(self) -> str:
                return "test"

            def _apply_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

        transformer_no_gpu = ConcreteTransformer(use_gpu=False)
        transformer_with_gpu = ConcreteTransformer(use_gpu=True)

        assert transformer_no_gpu.use_gpu is False
        assert transformer_with_gpu.use_gpu is True

    def it_returns_correct_engine_based_on_gpu_flag(self) -> None:
        """Verify _get_engine returns correct engine."""

        class ConcreteTransformer(BaseDataTransformer):
            @property
            def dataset_name(self) -> str:
                return "test"

            def _apply_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
                return df

        assert ConcreteTransformer(use_gpu=False)._get_engine() == "auto"
        assert ConcreteTransformer(use_gpu=True)._get_engine() == "gpu"


class DescribeTaiwanCreditTransformer:
    """Tests for TaiwanCreditTransformer."""

    @pytest.fixture
    def sample_taiwan_data(self) -> pl.DataFrame:
        """Create sample Taiwan Credit data."""
        return pl.DataFrame(
            {
                "ID": [1, 2, 3],
                "LIMIT_BAL": [20000, 30000, 40000],
                "SEX": [1, 2, 1],
                "EDUCATION": [1, 2, 5],  # 5 is "unknown", should become 4
                "MARRIAGE": [1, 2, 0],  # 0 is "unknown", should become 3
                "AGE": [25, 30, 35],
                "PAY_0": [-1, 0, 2],  # -1 and 0 should become 0
                "PAY_2": [0, 1, 3],
                "PAY_3": [-2, 0, 1],
                "PAY_4": [0, 0, 2],
                "PAY_5": [-1, 0, 0],
                "PAY_6": [0, 0, 1],
                "BILL_AMT1": [1000, 2000, 3000],
                "PAY_AMT1": [500, 1000, 1500],
                "default.payment.next.month": [0, 1, 0],
            }
        )

    def it_satisfies_data_transformer_protocol(self) -> None:
        """Verify transformer satisfies the protocol."""
        transformer = TaiwanCreditTransformer()
        assert isinstance(transformer, DataTransformer)

    def it_has_correct_dataset_name(self) -> None:
        """Verify dataset name property."""
        transformer = TaiwanCreditTransformer()
        assert transformer.dataset_name == "taiwan_credit"

    def it_renames_target_column(self, sample_taiwan_data: pl.DataFrame) -> None:
        """Verify target column is renamed from 'default.payment.next.month'."""
        transformer = TaiwanCreditTransformer()
        result = transformer.transform(sample_taiwan_data, Dataset.TAIWAN_CREDIT)

        assert "target" in result.columns
        assert "default.payment.next.month" not in result.columns

    def it_removes_id_column(self, sample_taiwan_data: pl.DataFrame) -> None:
        """Verify ID column is removed."""
        transformer = TaiwanCreditTransformer()
        result = transformer.transform(sample_taiwan_data, Dataset.TAIWAN_CREDIT)

        assert "ID" not in result.columns

    def it_normalizes_pay_columns(self, sample_taiwan_data: pl.DataFrame) -> None:
        """Verify PAY columns have negative values converted to 0."""
        transformer = TaiwanCreditTransformer()
        result = transformer.transform(sample_taiwan_data, Dataset.TAIWAN_CREDIT)

        # All PAY columns should have values >= 0
        pay_cols = [col for col in result.columns if col.startswith("PAY_") and col[4:].isdigit()]
        for col in pay_cols:
            assert result[col].min() >= 0  # type: ignore[operator]

    def it_creates_one_hot_encoded_columns(self, sample_taiwan_data: pl.DataFrame) -> None:
        """Verify categorical columns are one-hot encoded."""
        transformer = TaiwanCreditTransformer()
        result = transformer.transform(sample_taiwan_data, Dataset.TAIWAN_CREDIT)

        # Should have encoded columns for EDUCATION, MARRIAGE, SEX
        education_cols = [col for col in result.columns if col.startswith("EDUCATION_CAT")]
        marriage_cols = [col for col in result.columns if col.startswith("MARRIAGE_CAT")]
        sex_cols = [col for col in result.columns if col.startswith("SEX_CAT")]

        assert len(education_cols) > 0
        assert len(marriage_cols) > 0
        assert len(sex_cols) > 0


class DescribeLendingClubTransformer:
    """Tests for LendingClubTransformer."""

    @pytest.fixture
    def sample_lending_data(self) -> pl.DataFrame:
        """Create minimal sample Lending Club data."""
        return pl.DataFrame(
            {
                "loan_status": ["Fully Paid", "Charged Off", "Fully Paid"],
                "loan_amnt": [10000, 20000, 15000],
                "installment": [300.0, 600.0, 450.0],
                "annual_inc": [60000.0, 80000.0, 70000.0],
                "dti": [15.0, 25.0, 20.0],
                "delinq_2yrs": [0, 1, 0],
                "inq_last_6mths": [1, 2, 1],
                "open_acc": [5, 10, 7],
                "pub_rec": [0, 0, 1],
                "revol_bal": [5000, 10000, 7500],
                "revol_util": [30.0, 50.0, 40.0],
                "total_acc": [10, 20, 15],
                "avg_cur_bal": [10000, 15000, 12000],
                "total_rev_hi_lim": [20000, 30000, 25000],
                "acc_open_past_24mths": [2, 3, 2],
                "percent_bc_gt_75": [20.0, 40.0, 30.0],
                "inq_fi": [1, 2, 1],
                "emp_length": ["5 years", "10+ years", "< 1 year"],
                "issue_d": ["Jan-2020", "Feb-2020", "Mar-2020"],
                "earliest_cr_line": ["Jan-2010", "Jan-2005", "Jan-2015"],
                "term": [" 36 months", " 60 months", " 36 months"],
                "home_ownership": ["RENT", "OWN", "MORTGAGE"],
                "verification_status": ["Verified", "Not Verified", "Source Verified"],
                "purpose": ["debt_consolidation", "credit_card", "home_improvement"],
            }
        )

    def it_satisfies_data_transformer_protocol(self) -> None:
        """Verify transformer satisfies the protocol."""
        transformer = LendingClubTransformer()
        assert isinstance(transformer, DataTransformer)

    def it_has_correct_dataset_name(self) -> None:
        """Verify dataset name property."""
        transformer = LendingClubTransformer()
        assert transformer.dataset_name == "lending_club"

    def it_creates_binary_target(self, sample_lending_data: pl.DataFrame) -> None:
        """Verify target is binary (0 for Fully Paid, 1 for Charged Off)."""
        transformer = LendingClubTransformer()
        result = transformer.transform(sample_lending_data, Dataset.LENDING_CLUB)

        assert "target" in result.columns
        assert result["target"].to_list() == [0, 1, 0]

    def it_filters_out_non_final_statuses(self) -> None:
        """Verify only Fully Paid and Charged Off are kept."""
        data = pl.DataFrame(
            {
                "loan_status": ["Current", "Fully Paid", "Charged Off", "In Grace Period"],
                "loan_amnt": [10000, 20000, 15000, 12000],
                "installment": [300.0, 600.0, 450.0, 350.0],
                "annual_inc": [60000.0, 80000.0, 70000.0, 65000.0],
                "dti": [15.0, 25.0, 20.0, 18.0],
                "delinq_2yrs": [0, 1, 0, 0],
                "inq_last_6mths": [1, 2, 1, 1],
                "open_acc": [5, 10, 7, 6],
                "pub_rec": [0, 0, 1, 0],
                "revol_bal": [5000, 10000, 7500, 6000],
                "revol_util": [30.0, 50.0, 40.0, 35.0],
                "total_acc": [10, 20, 15, 12],
                "avg_cur_bal": [10000, 15000, 12000, 11000],
                "total_rev_hi_lim": [20000, 30000, 25000, 22000],
                "acc_open_past_24mths": [2, 3, 2, 2],
                "percent_bc_gt_75": [20.0, 40.0, 30.0, 25.0],
                "inq_fi": [1, 2, 1, 1],
                "emp_length": ["5 years", "10+ years", "< 1 year", "3 years"],
                "issue_d": ["Jan-2020", "Feb-2020", "Mar-2020", "Apr-2020"],
                "earliest_cr_line": ["Jan-2010", "Jan-2005", "Jan-2015", "Jan-2012"],
                "term": [" 36 months", " 60 months", " 36 months", " 36 months"],
                "home_ownership": ["RENT", "OWN", "MORTGAGE", "RENT"],
                "verification_status": ["Verified", "Not Verified", "Source Verified", "Verified"],
                "purpose": ["debt_consolidation", "credit_card", "home_improvement", "other"],
            }
        )

        transformer = LendingClubTransformer()
        result = transformer.transform(data, Dataset.LENDING_CLUB)

        # Only 2 rows should remain (Fully Paid and Charged Off)
        assert len(result) == 2


class DescribeCorporateCreditTransformer:
    """Tests for CorporateCreditTransformer."""

    @pytest.fixture
    def sample_corporate_data(self) -> pl.DataFrame:
        """Create sample Corporate Credit data."""
        return pl.DataFrame(
            {
                "Name": ["Company A", "Company B", "Company C"],
                "Symbol": ["AAA", "BBB", "CCC"],
                "Rating Agency Name": ["S&P", "Moody's", "Fitch"],
                "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "Rating": ["AAA", "BBB", "D"],  # D should be target=1
                "Sector": ["Technology", "Finance", "Healthcare"],
                "currentRatio": [1.5, 2.0, 1.0],
                "quickRatio": [1.2, 1.8, 0.8],
                "cashRatio": [0.5, 1.0, 0.3],
                "daysOfSalesOutstanding": [30.0, 45.0, 60.0],
            }
        )

    def it_satisfies_data_transformer_protocol(self) -> None:
        """Verify transformer satisfies the protocol."""
        transformer = CorporateCreditTransformer()
        assert isinstance(transformer, DataTransformer)

    def it_has_correct_dataset_name(self) -> None:
        """Verify dataset name property."""
        transformer = CorporateCreditTransformer()
        assert transformer.dataset_name == "corporate_credit"

    def it_creates_binary_target_from_rating(self, sample_corporate_data: pl.DataFrame) -> None:
        """Verify 'D' rating becomes target=1, others become 0."""
        transformer = CorporateCreditTransformer()
        result = transformer.transform(sample_corporate_data, Dataset.CORPORATE_CREDIT_RATING)

        assert "target" in result.columns
        # Third row has Rating='D', should be 1
        assert result["target"].to_list() == [0, 0, 1]

    def it_removes_metadata_columns(self, sample_corporate_data: pl.DataFrame) -> None:
        """Verify non-predictive columns are removed."""
        transformer = CorporateCreditTransformer()
        result = transformer.transform(sample_corporate_data, Dataset.CORPORATE_CREDIT_RATING)

        assert "Name" not in result.columns
        assert "Symbol" not in result.columns
        assert "Rating Agency Name" not in result.columns
        assert "Date" not in result.columns
        assert "Rating" not in result.columns

    def it_one_hot_encodes_sector(self, sample_corporate_data: pl.DataFrame) -> None:
        """Verify Sector is one-hot encoded."""
        transformer = CorporateCreditTransformer()
        result = transformer.transform(sample_corporate_data, Dataset.CORPORATE_CREDIT_RATING)

        # Original Sector column should be gone
        assert "Sector" not in result.columns

        # Should have one-hot encoded columns
        sector_cols = [col for col in result.columns if col.startswith("Sector_")]
        assert len(sector_cols) == 3  # Technology, Finance, Healthcare

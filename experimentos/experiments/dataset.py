import re

from loguru import logger
import polars as pl
from polars import datatypes
import typer
from typing_extensions import Annotated

from experiments.config import Dataset

app = typer.Typer()


@Dataset.LENDING_CLUB.register_dataset_processor()
def preprocess_lending_club_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Processa o conjunto de dados do Lending Club com base na metodologia de
    Namvar et al. (2018).

    Isso inclui:
    1.  Binarização da variável-alvo ('loan_status').
    2.  Engenharia de features (cálculo de 'credit_age', 'new_dti', etc.).
    3.  Limpeza e transformação de colunas (ex: 'emp_length').
    4.  Transformação de log em features assimétricas.
    5.  One-hot encoding das features categóricas.
    6.  Seleção final das features para o modelo, removendo dados vazados (leaky data).

    Args:
        df: O DataFrame Polars bruto do Lending Club.

    Returns:
        Um DataFrame Polars limpo e pronto para o pipeline de ML.
    """

    # --- 1. Definição da Variável Alvo (Target) ---
    # Com base na Seção 4.1, "current" loans são removidos.
    # O alvo é binário: 'Charged Off' (1) vs 'Fully Paid' (0).
    target_map = {"Charged Off": 1, "Fully Paid": 0}

    # --- 2. Início do Pré-processamento (Lazy) ---
    df_processed = (
        df.lazy()
        .filter(pl.col("loan_status").is_in(target_map.keys()))
        .with_columns(
            # Criar a coluna alvo 'target'
            pl.when(pl.col("loan_status") == "Charged Off").then(1).otherwise(0).alias("target"),
            # Converter datas de string para datetime para cálculo
            pl.col("issue_d").str.to_date(format="%b-%Y", strict=False).alias("issue_d_dt"),
            pl.col("earliest_cr_line")
            .str.to_date(format="%b-%Y", strict=False)
            .alias("earliest_cr_line_dt"),
            # Converter 'emp_length' para numérico
            pl.col("emp_length").str.extract(r"(\d+)", 1).cast(pl.Float64).alias("emp_length_num"),
        )
        .with_columns(
            # --- 3. Engenharia de Features (Baseada na Tabela 1 e Seção 4.1) ---
            # Calcular 'credit_age'
            # (idade do crédito em meses, da abertura da 1ª linha até a emissão do empréstimo)
            (
                (pl.col("issue_d_dt") - pl.col("earliest_cr_line_dt")).dt.total_days() / 30.4375
            ).alias("credit_age_months"),
            # Calcular 'monthly_inc' para usar nos ratios
            (pl.col("annual_inc") / 12).alias("monthly_inc"),
        )
        .with_columns(
            # Calcular Ratios Derivados [cite: 1999-2007]
            # 'income_to_payment_ratio' [cite: 1999]
            (pl.col("monthly_inc") / pl.col("installment")).alias("income_to_payment_ratio"),
            # 'revolving_to_income_ratio' [cite: 1995]
            (pl.col("revol_bal") / pl.col("monthly_inc")).alias("revol_to_income_ratio"),
            # 'new_dti' (New Debt-to-Income) [cite: 2000-2007]
            # NMRA = New Monthly Repayment Amount
            (
                ((pl.col("dti") * pl.col("monthly_inc")) + pl.col("installment"))
                / pl.col("monthly_inc")
            ).alias("new_dti"),
        )
        .with_columns(
            # Tratar divisões por zero que resultam em infinito apenas nas colunas float
            pl.col(datatypes.Float32).replace([float("inf"), float("-inf")], None)
        )
        .with_columns(
            # --- 4. Transformações de Log ---
            # Aplicar log1p (log(x+1)) em features assimétricas
            pl.col("annual_inc").fill_null(0).log1p(),
            # Preenche nulos com 0 antes do log. Nulos aqui significam
            # provável divisão por 0 (ex: sem 'installment' ou 'monthly_inc')
            pl.col("income_to_payment_ratio").fill_null(0).log1p(),
            pl.col("revol_to_income_ratio").fill_null(0).log1p(),
        )
    )

    # --- 5. Seleção Final de Features ---
    # Com base na Tabela 1 do artigo

    # Features numéricas a manter (incluindo as que criamos)
    numeric_cols = [
        "loan_amnt",
        "installment",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "avg_cur_bal",
        "total_rev_hi_lim",
        "acc_open_past_24mths",
        "percent_bc_gt_75",
        "inq_fi",
        "emp_length_num",  # Variável tratada
        "credit_age_months",  # Variável de idade do crédito
        "new_dti",  # Variável derivada [cite: 2000]
        "income_to_payment_ratio",  # Variável derivada [cite: 1999]
        "revol_to_income_ratio",  # Variável derivada [cite: 1995]
    ]

    # Features categóricas a manter (para one-hot encoding)
    categorical_cols = ["term", "home_ownership", "verification_status", "purpose"]

    target_col = ["target"]

    # Selecionar apenas as colunas de interesse
    final_features_df = df_processed.select(numeric_cols + categorical_cols + target_col)

    # --- 6. One-Hot Encoding ---
    # Converter as colunas categóricas em colunas dummy (binárias)

    final_df_with_dummies = final_features_df.collect().to_dummies(
        columns=categorical_cols,
        separator="_",
        drop_first=False,  # Mantém todas as categorias
    )

    # Substituir caracteres inválidos nos nomes das colunas (para o Scikit-learn)
    final_df_with_dummies.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "", col) for col in final_df_with_dummies.columns
    ]

    return final_df_with_dummies


@Dataset.CORPORATE_CREDIT_RATING.register_dataset_processor()
def preprocess_corporate_credit_data(raw_data: pl.DataFrame) -> pl.DataFrame:
    """
    Pré-processa o conjunto de dados de Crédito Corporativo.

    Esta função realiza as seguintes etapas com base na metodologia do PGC:
    1.  Binariza a variável-alvo 'Rating':
        - 'D' (maior risco) torna-se 1 (classe positiva/minoritária).
        - Todas as outras ('AAA' a 'C') tornam-se 0 (classe negativa/majoritária).
    2.  Remove colunas de metadados não preditivas (ex: Name, Symbol, Date).
    3.  Faz One-hot encoding da feature categórica 'Sector'.
    4.  Seleciona todos os indicadores financeiros (Float64) como features.

    Args:
        df: O DataFrame Polars bruto de crédito corporativo.

    Returns:
        Um DataFrame Polars limpo e pronto para o pipeline de ML.
    """

    # --- 1. Definição da Variável Alvo (Target) ---
    # Conforme Seção 3.2 do capitulo/ferramentas.tex
    df_processed = raw_data.lazy().with_columns(
        pl.when(pl.col("Rating") == "D").then(pl.lit(1)).otherwise(pl.lit(0)).alias("target")
    )

    # --- 2. Seleção de Features ---

    # Identificar automaticamente todas as colunas numéricas (Float64)
    # que são os indicadores fundamentalistas.
    numeric_cols = [col for col, dtype in raw_data.schema.items() if dtype == pl.Float64]

    # Coluna categórica a ser codificada
    categorical_cols = ["Sector"]

    # Coluna alvo
    target_col = ["target"]

    # Selecionar apenas as colunas de interesse
    # Isso descarta 'Rating' (original), 'Name', 'Symbol', 'Rating Agency Name', 'Date'
    final_features_df = df_processed.select(numeric_cols + categorical_cols + target_col)

    # --- 3. One-Hot Encoding da feature 'Sector' ---
    # Coletamos o resultado (executamos o plano lazy) e aplicamos o to_dummies
    final_df_with_dummies = final_features_df.collect().to_dummies(
        columns=categorical_cols,
        separator="_",
        drop_first=False,  # Mantém todas as categorias
    )

    # Limpa nomes de colunas para compatibilidade (remove caracteres especiais)
    final_df_with_dummies.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "", col) for col in final_df_with_dummies.columns
    ]

    return final_df_with_dummies


@Dataset.TAIWAN_CREDIT.register_dataset_processor()
def preprocess_taiwan_credit_data(raw_data: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocesses the Taiwan Credit Card dataset.

    This function performs the following steps based on the dataset description
    and the PGC methodology:
    1.  Renames the target variable 'default.payment.next.month' to 'target'
        for consistency.
    2.  Removes the 'ID' column as it is not predictive.
    3.  Cleans and groups the categorical features 'EDUCATION' and 'MARRIAGE'
        to remove "unknown" or "other" values and simplify.
    4.  Maps 'SEX' to strings ('Male', 'Female') for clarity.
    5.  Normalizes the columns 'PAY_0' to 'PAY_6', treating "on-time payment"
        (values <= 0) as 0 and keeping the months of delay (1-9).
    6.  Converts the cleaned categorical columns to one-hot encoding.

    Args:
        df: The raw Polars DataFrame of Taiwan Credit.

    Returns:
        A clean Polars DataFrame ready for the ML pipeline.
    """

    # --- 1. Definition of Mappings ---

    # Payment status columns
    pay_cols = [
        col
        for col in raw_data.columns
        if any([col.startswith("PAY_"), col.startswith("BILL_"), col.startswith("PAY_AMT_")])
    ]

    # --- 2. Start of Preprocessing (Lazy) ---
    df_processed = (
        raw_data.lazy()
        .with_columns(
            # Rename target for consistency
            pl.col("default.payment.next.month").alias("target"),
            # Clean EDUCATION
            pl.when(pl.col("EDUCATION") <= 3)
            .then(pl.col("EDUCATION"))
            .otherwise(4)
            .cast(pl.String)  # Convert to string before to_dummies
            .alias("EDUCATION_CAT"),
            # Clean MARRIAGE
            pl.when(pl.col("MARRIAGE") <= 2)
            .then(pl.col("MARRIAGE"))
            .otherwise(3)
            .cast(pl.String)
            .alias("MARRIAGE_CAT"),
            # Map SEX
            pl.col("SEX").cast(pl.String).alias("SEX_CAT"),
        )
        .with_columns(
            [
                # Clean PAY_* columns:
                # Maps values <= 0 (on-time payment, early, etc.) to 0.
                # Keeps 1-9 (months of delay)
                pl.when(pl.col(c) <= 0).then(pl.lit(0)).otherwise(pl.col(c)).alias(c)
                for c in pay_cols
            ]
        )
    )

    # --- 3. Final Selection and One-Hot Encoding ---

    # Categorical columns we just created
    categorical_to_encode = ["EDUCATION_CAT", "MARRIAGE_CAT", "SEX_CAT"]

    # Original columns to drop
    cols_to_drop = ["ID", "default.payment.next.month", "EDUCATION", "MARRIAGE", "SEX"]

    # Collect (execute) the lazy plan and apply to_dummies
    final_df_with_dummies = (
        df_processed.drop(cols_to_drop)
        .collect()
        .to_dummies(columns=categorical_to_encode, separator="_", drop_first=False)
    )

    # Clean column names for compatibility
    final_df_with_dummies.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "", col) for col in final_df_with_dummies.columns
    ]

    return final_df_with_dummies


@app.command(name="process")
def main(
    dataset: Annotated[
        Dataset,
        typer.Argument(..., help="The identifier of the dataset to process."),
    ],
):
    """Processes the specified dataset and saves the result."""
    logger.info(f"Processing dataset {dataset}...")

    raw_data_path = dataset.get_raw_data_path()
    processed_data_path = dataset.get_processed_data_path()

    logger.info(f"Loading raw data from {raw_data_path}...")
    raw_data = pl.read_csv(raw_data_path, **dataset.get_extra_params())
    logger.info("Raw data loaded.")

    processed_data = dataset.process_data(raw_data)

    logger.info(f"Saving processed data to {processed_data_path}...")
    processed_data.write_parquet(processed_data_path)

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()

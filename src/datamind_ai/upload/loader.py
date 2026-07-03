from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from datamind_ai.config import MAX_FILE_SIZE_MB, SUPPORTED_EXTENSIONS


@dataclass
class UploadResult:
    success: bool
    df: pd.DataFrame | None = None
    filename: str = ""
    error: str | None = None
    sheet_name: str | None = None


@dataclass
class MultiUploadResult:
    success: bool
    datasets: list[UploadResult]
    filename: str = ""
    error: str | None = None


def file_fingerprint(filename: str, content: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(filename.encode("utf-8"))
    digest.update(content)
    return digest.hexdigest()


def _validate_file(filename: str, content: bytes) -> str | None:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return (
            f"Formato não suportado: '{ext}'. "
            f"Formatos aceites: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if len(content) == 0:
        return "O ficheiro está vazio."

    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return (
            f"O ficheiro excede o limite de {MAX_FILE_SIZE_MB} MB "
            f"({size_mb:.1f} MB)."
        )

    return None


def _load_csv(content: bytes) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Não foi possível determinar a codificação do ficheiro CSV.")


def _load_excel(content: bytes, sheet_name: str | None = None) -> pd.DataFrame:
    return pd.read_excel(
        io.BytesIO(content), sheet_name=sheet_name, engine="openpyxl"
    )


def list_excel_sheets(content: bytes) -> list[str]:
    xl = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
    return xl.sheet_names


def _load_json(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return pd.DataFrame(data["data"])
        return pd.DataFrame([data])
    raise ValueError("O JSON deve ser uma lista de objetos ou um objeto com chave 'data'.")


def _validate_dataframe(df: pd.DataFrame, filename: str) -> UploadResult | None:
    if df.empty or len(df.columns) == 0:
        return UploadResult(
            success=False,
            filename=filename,
            error="O ficheiro não contém dados (sem linhas ou sem colunas).",
        )
    return None


def load_file(
    filename: str, content: bytes, sheet_name: str | None = None
) -> UploadResult | MultiUploadResult:
    """Load a file; Excel with multiple sheets returns one dataset per sheet."""
    validation_error = _validate_file(filename, content)
    if validation_error:
        return UploadResult(success=False, filename=filename, error=validation_error)

    ext = Path(filename).suffix.lower()
    try:
        if ext == ".csv":
            df = _load_csv(content)
            empty = _validate_dataframe(df, filename)
            if empty:
                return empty
            return UploadResult(success=True, df=df, filename=filename)
        if ext == ".xlsx":
            sheets = list_excel_sheets(content)
            if len(sheets) == 1:
                df = _load_excel(content, sheets[0])
                empty = _validate_dataframe(df, filename)
                if empty:
                    return empty
                return UploadResult(
                    success=True, df=df, filename=filename, sheet_name=sheets[0]
                )
            results = []
            for sheet in sheets:
                df = _load_excel(content, sheet)
                if df.empty or len(df.columns) == 0:
                    continue
                results.append(
                    UploadResult(
                        success=True,
                        df=df,
                        filename=filename,
                        sheet_name=sheet,
                    )
                )
            if not results:
                return UploadResult(
                    success=False,
                    filename=filename,
                    error="O Excel não contém folhas com dados.",
                )
            return MultiUploadResult(success=True, datasets=results, filename=filename)
        if ext == ".json":
            df = _load_json(content)
            empty = _validate_dataframe(df, filename)
            if empty:
                return empty
            return UploadResult(success=True, df=df, filename=filename)
        return UploadResult(
            success=False, filename=filename, error=f"Formato não suportado: {ext}"
        )
    except json.JSONDecodeError:
        return UploadResult(
            success=False,
            filename=filename,
            error="Ficheiro JSON inválido ou corrompido.",
        )
    except Exception as exc:
        return UploadResult(
            success=False,
            filename=filename,
            error=f"Erro ao processar o ficheiro: {exc}",
        )


def load_dataset(filename: str, content: bytes) -> UploadResult:
    """Backward-compatible single-dataset loader."""
    result = load_file(filename, content)
    if isinstance(result, MultiUploadResult):
        first = result.datasets[0]
        return UploadResult(
            success=first.success,
            df=first.df,
            filename=filename,
            error=None if first.success else "Sem dados",
            sheet_name=first.sheet_name,
        )
    return result


def make_dataset_name(
    filename: str,
    existing: set[str],
    sheet_name: str | None = None,
) -> str:
    base = Path(filename).stem
    if sheet_name:
        safe_sheet = sheet_name.replace(" ", "_")
        base = f"{base}_{safe_sheet}"
    name = base
    counter = 1
    while name in existing:
        name = f"{base}_{counter}"
        counter += 1
    return name

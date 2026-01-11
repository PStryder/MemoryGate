import os

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("VECTOR_BACKEND", "none")


def test_import_harness():
    import core.import_harness  # noqa: F401

## Development Guide

### Running Tests

- **Install dependencies** (once per environment):

```bash
pip install -r requirements.txt
```

- **Start a local Postgres for testing** (Docker, Mac/Linux/Windows with Docker):

```bash
docker run --name backend-test-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=audiodb \
  -p 5432:5432 \
  -d postgres:16
```

- **Run all tests**:

```bash
PYTHONPATH=. pytest
```

- **Run Postgres-backed tests as well**  
  (requires a running Postgres instance; locally we typically point to `localhost:5432`):

```bash
# Mac / Linux
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/audiodb"
export TEST_WITH_DB=1
PYTHONPATH=. pytest tests/test_auth_postgres_integration.py -q

# Windows (cmd)
set DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/audiodb
set TEST_WITH_DB=1
PYTHONPATH=. pytest tests/test_auth_postgres_integration.py -q

# Windows (PowerShell)
$env:DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/audiodb"
$env:TEST_WITH_DB = "1"
PYTHONPATH=. pytest tests/test_auth_postgres_integration.py -q
```


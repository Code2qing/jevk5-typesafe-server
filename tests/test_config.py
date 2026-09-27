import os
from jevk5_server.config import Settings
from run import parse_args, build_settings


def test_env_var_respected_when_no_cli_args(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://my-remote-llama:9090")
    monkeypatch.setenv("MODEL_NAME", "custom-9b-model")
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("TEMPERATURE", "1.05")

    args = parse_args([])
    settings = build_settings(args)

    assert settings.llama_server_url == "http://my-remote-llama:9090"
    assert settings.model_name == "custom-9b-model"
    assert settings.port == 9999
    assert settings.temperature == 1.05


def test_cli_args_override_env_vars(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://env-url:8080")

    args = parse_args(["--llama-url", "http://cli-override:8080"])
    settings = build_settings(args)

    assert settings.llama_server_url == "http://cli-override:8080"


def test_default_used_when_neither_env_nor_cli_provided(monkeypatch):
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)

    args = parse_args([])
    settings = build_settings(args)

    assert settings.llama_server_url == "http://127.0.0.1:8080"


def test_settings_reads_from_dotenv(tmp_path, monkeypatch):
    # Create a temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text("LLAMA_SERVER_URL=http://from-dotenv:8888\nPORT=7777\nAPI_KEY=env-secret-key\n")

    # Clear any environment variables that might interfere
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    # Change directory to tmp_path where .env resides
    monkeypatch.chdir(tmp_path)

    settings = Settings()
    assert settings.llama_server_url == "http://from-dotenv:8888"
    assert settings.port == 7777
    assert settings.api_key == "env-secret-key"


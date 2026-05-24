from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import threading
import time
from typing import Optional

import click
import httpx
import uvicorn

from inferr.config import Config, load_config
from inferr import server as server_module


@click.group()
def cli() -> None:
    """Inferr command line interface."""


def _resolve_port(config: Config, port_override: Optional[int]) -> int:
    if port_override is not None:
        return port_override
    env_port = os.environ.get("INFERR_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            return config.port
    return config.port


def _resolve_host(config: Config, host_override: Optional[str]) -> str:
    if host_override is not None:
        return host_override
    return config.host


def _start_session_when_ready(base_url: str) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=1.0) as client:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    break
        except httpx.ConnectError:
            time.sleep(0.2)
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        return

    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(f"{base_url}/session/start")
    except httpx.HTTPError:
        return


@cli.command()
@click.option("--host", type=str, default=None)
@click.option("--port", type=int, default=None)
@click.option("--no-browser", is_flag=True, default=False)
@click.option("--lang", type=str, default=None)
def start(
    host: Optional[str], port: Optional[int], no_browser: bool, lang: Optional[str]
) -> None:
    """Start the Inferr server and initialize a session."""
    config = load_config()
    if not config.gemini.api_key and not config.groq.api_key:
        click.echo(
            "[inferr] ERROR: No LLM API key set. "
            "Add GROQ_API_KEY or GEMINI_API_KEY to your .env file.",
            err=True,
        )
        raise SystemExit(1)
    resolved_port = _resolve_port(config, port)
    resolved_host = _resolve_host(config, host)
    resolved_lang = lang if lang is not None else config.language

    config = replace(
        config, host=resolved_host, port=resolved_port, language=resolved_lang
    )

    server_module.CONFIG_OVERRIDE = config
    os.environ["INFERR_PORT"] = str(resolved_port)
    if no_browser:
        os.environ["INFERR_NO_BROWSER"] = "1"

    base_url = f"http://{resolved_host}:{resolved_port}"
    thread = threading.Thread(
        target=_start_session_when_ready, args=(base_url,), daemon=True
    )
    thread.start()

    uvicorn.run(
        "inferr.server:app", host=resolved_host, port=resolved_port, log_level="info"
    )


@cli.command()
def stop() -> None:
    """Stop the current Inferr session."""
    config = load_config()
    port_value = _resolve_port(config, None)
    base_url = f"http://{config.host}:{port_value}"

    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(f"{base_url}/session/stop")
        click.echo("Inferr session stopped.")
    except httpx.ConnectError:
        click.echo("Inferr is not running. Start it with `inferr start`.")
    except httpx.HTTPError:
        click.echo("Failed to stop session.")


@cli.command()
def status() -> None:
    """Show the current Inferr status."""
    config = load_config()
    port_value = _resolve_port(config, None)
    base_url = f"http://{config.host}:{port_value}"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/health")
        data = response.json()
        click.echo(f"Session active: {data.get('session_active')}")
        click.echo(f"Last activity: {data.get('last_activity')}")
    except httpx.ConnectError:
        click.echo("Inferr is not running. Start it with `inferr start`.")
    except httpx.HTTPError:
        click.echo("Failed to fetch status.")


@cli.command()
@click.option("--n", type=int, default=None)
def logs(n: Optional[int]) -> None:
    """Show recent terminal buffer lines and conversation history."""
    config = load_config()
    port_value = _resolve_port(config, None)
    base_url = f"http://{config.host}:{port_value}"
    limit = n if n is not None else config.terminal_buffer_lines

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/context")
        if response.status_code == 404:
            click.echo("No active session.")
            return
        data = response.json()
        terminal_lines = data.get("terminal_buffer", [])[-limit:]
        history = data.get("conversation_history", [])

        click.echo("--- Terminal Buffer ---")
        for line in terminal_lines:
            click.echo(line)

        click.echo("--- Conversation History ---")
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            click.echo(f"{role}: {content}")
    except httpx.ConnectError:
        click.echo("Inferr is not running. Start it with `inferr start`.")
    except httpx.HTTPError:
        click.echo("Failed to fetch logs.")


def _debug_base_url() -> str:
    config = load_config()
    port_value = _resolve_port(config, None)
    return f"http://{config.host}:{port_value}"


@cli.group()
def debug() -> None:
    """Pipeline isolation tests — one subsystem at a time."""


@debug.command("llm")
@click.argument("text", default="hello")
def debug_llm(text: str) -> None:
    """Test Groq LLM only (no STT, TTS, or websocket)."""
    base_url = _debug_base_url()
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{base_url}/debug/pipeline/llm",
                json={"transcript": text},
            )
        data = response.json()
        stage = data.get("pipeline_stage", {})
        click.echo(f"PIPELINE_{stage.get('stage', '?')}: {'OK' if stage.get('ok') else 'FAILED'}")
        if stage.get("detail"):
            click.echo(f"detail: {stage.get('detail')}")
        if response.status_code == 200:
            click.echo(f"response: {data.get('response', '')}")
        else:
            click.echo(f"error: {data.get('error', data)}", err=True)
            raise SystemExit(1)
    except httpx.ConnectError:
        click.echo("Inferr is not running. Start it with `inferr start`.", err=True)
        raise SystemExit(1)


@debug.command("tts")
@click.argument("text", default="hello from inferr")
@click.option("--tone", default="neutral")
def debug_tts(text: str, tone: str) -> None:
    """Test TTS only. Silk needs the browser UI open on /ws."""
    base_url = _debug_base_url()
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{base_url}/debug/pipeline/tts",
                json={"text": text, "tone": tone},
            )
        data = response.json()
        click.echo(f"status: {data.get('status')}")
        if data.get("detail"):
            click.echo(f"detail: {data.get('detail')}")
        if data.get("hint"):
            click.echo(f"hint: {data.get('hint')}")
        if response.status_code != 200:
            raise SystemExit(1)
    except httpx.ConnectError:
        click.echo("Inferr is not running. Start it with `inferr start`.", err=True)
        raise SystemExit(1)


@debug.command("status")
def debug_status() -> None:
    """Show which pipeline stages can run right now."""
    base_url = _debug_base_url()
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/debug/pipeline/status")
        for key, value in response.json().items():
            click.echo(f"{key}: {value}")
    except httpx.ConnectError:
        click.echo("Inferr is not running.", err=True)
        raise SystemExit(1)


@cli.command("install-shell")
@click.option("--shell", "shell_name", type=click.Choice(["zsh", "bash"]), default=None)
def install_shell(shell_name: Optional[str]) -> None:
    """Install the Inferr shell integration plugin."""
    import shutil

    detected = os.environ.get("SHELL", "")
    if shell_name is None:
        if "zsh" in detected:
            shell_name = "zsh"
        elif "bash" in detected:
            shell_name = "bash"
        else:
            click.echo("Could not detect shell. Use --shell zsh or --shell bash.")
            raise SystemExit(1)

    plugin_src = Path(__file__).resolve().parent / "shell" / f"inferr.{shell_name}"
    if not plugin_src.exists():
        click.echo(f"Plugin file not found: {plugin_src}")
        raise SystemExit(1)

    inferr_dir = Path.home() / ".inferr"
    inferr_dir.mkdir(parents=True, exist_ok=True)
    dest = inferr_dir / f"inferr.{shell_name}"
    shutil.copy(plugin_src, dest)

    rc_file = Path.home() / (f".{shell_name}rc")
    source_line = f'\nsource "{dest}"  # inferr shell integration\n'

    if rc_file.exists():
        content = rc_file.read_text(encoding="utf-8")
        if str(dest) in content:
            click.echo(f"Already installed in {rc_file}.")
            return
        rc_file.write_text(content + source_line, encoding="utf-8")
    else:
        rc_file.write_text(source_line, encoding="utf-8")

    click.echo(f"Installed inferr shell plugin to {rc_file}.")
    click.echo(f"Run: source {rc_file}")

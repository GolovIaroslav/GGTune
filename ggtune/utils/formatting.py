from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def panel(text: str, title: str = "", style: str = "cyan") -> Panel:
    return Panel(text, title=title, border_style=style)


def success(msg: str) -> None:
    console.print(f"[bold green]✓[/] {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]⚠[/] {msg}")


def error(msg: str) -> None:
    console.print(f"[bold red]✗[/] {msg}")


def info(msg: str) -> None:
    console.print(f"[dim]→[/] {msg}")

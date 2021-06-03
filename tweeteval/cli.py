import typer
from .eval import published_results

CLI = typer.Typer()


@CLI.command()
def published():
    """ """
    typer.echo("Published:")
    print(published_results())

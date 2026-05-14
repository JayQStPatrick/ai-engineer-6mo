import click
from rich.console import Console
from rich.table import Table
from rag.ingest import load_document
from rag.chunker import chunk_recursive, chunk_token
from rag.embeddings import build_index
from rag.retriever import retrieve

console = Console()

@click.group()
def cli():
    """doc-ingest: load, chunk, embed, and search documents."""
    pass

@cli.command()
@click.argument("path")
@click.option("--strategy", default="recursive",
              type=click.Choice(["recursive", "token"]))
@click.option("--chunk-size", default=500)
@click.option("--overlap", default=50)
def ingest(path, strategy, chunk_size, overlap):
    """Load a document, chunk it, and build the search index."""
    console.print(f"Loading [bold]{path}[/bold]...")
    pages = load_document(path)
    console.print(f"  {len(pages)} pages loaded")

    if strategy == "recursive":
        chunks = chunk_recursive(pages, chunk_size, overlap)
    else:
        chunks = chunk_token(pages, chunk_size, overlap)
    console.print(f"  {len(chunks)} chunks created ({strategy})")

    build_index(chunks)
    console
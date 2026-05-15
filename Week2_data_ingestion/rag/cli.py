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
    console.print("[green]Index built.[/green]")

@cli.command()
@click.argument("query")
@click.option("--top-k", default=5)
def search(query, top_k):
    """Search the index for a query."""
    results = retrieve(query, top_k)
    table = Table(title=f"Top {top_k} results for: {query}")
    table.add_column("Citation", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Preview")
    for r in results:
        table.add_row(r["citation"], f"{r['score']:.3f}",
                      r["text"][:120] + "...")
    console.print(table)

if __name__ == "__main__":
    cli()
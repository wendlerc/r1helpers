from rich.console import Console
from rich.panel import Panel
import json

def print_response(all, new):
    # print the new part of the response in a different color (its the last part) and use the console to print it nicely
    console = Console(width=100)
    old_text = all[:-len(new)]
    new_text = all[-len(new):]
    console.print(Panel(f"[white]{old_text}[/white][bold white]{new_text}[/bold white]"))



def pprint(text):
    """Pretty print the model's generated text using rich."""
    console = Console(width=100)
    console.print(Panel(text, border_style="blue"))


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
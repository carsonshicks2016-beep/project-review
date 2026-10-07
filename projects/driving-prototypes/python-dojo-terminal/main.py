import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import print as rprint
from rich.prompt import Confirm
from prompt_toolkit import prompt
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.python import PythonLexer

from state_manager import load_progress, save_progress, update_progress_on_success, update_progress_on_failure, PROGRESS_FILE
from llm_engine import pull_model_if_missing, generate_exercise, generate_hint, generate_lesson, DEFAULT_MODEL
from execution_engine import execute_and_test

console = Console()

def display_dashboard(progress):
    """Displays the user's current progress."""
    dashboard = f"""
**Level:** {progress['level']}  
**Current Topic:** {progress['concept']}  
**Streak:** 🔥 {progress['streak']}  
**Total Solved:** {progress['total_solved']}
"""
    console.print(Panel(Markdown(dashboard), title="🥋 Python Dojo Progress 🥋", style="cyan"))

def main_loop():
    console.print("[bold green]Welcome to the Python Dojo![/bold green]")
    
    with console.status(f"[bold yellow]Checking for local model '{DEFAULT_MODEL}'..."):
        try:
            pull_model_if_missing()
        except Exception as e:
            console.print(f"[bold red]Failed to connect to Ollama. Make sure the Ollama app is running locally.[/bold red]")
            console.print(f"Error: {e}")
            sys.exit(1)

    progress = load_progress()
    
    if progress['total_solved'] > 0:
        console.print(f"\n[bold cyan]Found saved progress: Level {progress['level']} - {progress['concept']}[/bold cyan]")
        if not Confirm.ask("Would you like to continue from your save?"):
            from state_manager import DEFAULT_PROGRESS
            progress = DEFAULT_PROGRESS.copy()
            save_progress(progress)
            console.print("[bold yellow]Starting from scratch![/bold yellow]\n")

    while True:
        display_dashboard(progress)
        
        if progress['concept'] not in progress['seen_concepts']:
            with console.status("[bold cyan]The sensei is preparing a new lesson...[/bold cyan]"):
                lesson = generate_lesson(progress['concept'])
            console.print("\n")
            console.print(Panel(Markdown(lesson), title=f"🎓 New Concept: {progress['concept']} 🎓", border_style="blue"))
            progress['seen_concepts'].append(progress['concept'])
            save_progress(progress)
            
            console.print("\n[bold dim]Press Enter when you are ready to try an exercise![/bold dim]")
            try:
                prompt()
            except (KeyboardInterrupt, EOFError):
                break

        console.print(f"\n[bold yellow]Generating an exercise about '{progress['concept']}'...[/bold yellow]")
        
        with console.status("[bold cyan]Consulting the sensei (Ollama)...[/bold cyan]"):
            exercise = generate_exercise(progress['concept'], progress['level'])
            
        if not exercise:
            console.print("[bold red]Failed to generate an exercise. Please try again.[/bold red]")
            break
            
        # Display the exercise
        console.print("\n")
        console.print(Panel(
            Markdown(exercise.description), 
            title=f"Challenge: {exercise.concept}", 
            border_style="yellow"
        ))
        
        if exercise.starter_code:
            console.print("[bold dim]Starter Code:[/bold dim]")
            syntax = Syntax(exercise.starter_code, "python", theme="monokai", line_numbers=False)
            console.print(syntax)
            console.print("\n")

        current_code = exercise.starter_code + "\n" if exercise.starter_code else ""
        
        while True:
            # Prompt for user code
            console.print("[bold dim]Write your solution below. Press [Esc] then [Enter] to submit. (Or Ctrl+C to quit)[/bold dim]")
            try:
                user_code = prompt(
                    '>>> ',
                    multiline=True,
                    lexer=PygmentsLexer(PythonLexer),
                    default=current_code
                )
            except KeyboardInterrupt:
                console.print("\n[bold red]Exiting the Dojo. Goodbye![/bold red]")
                return
            except EOFError:
                return

            # Execute and Test
            console.print("\n[bold yellow]Evaluating your code...[/bold yellow]")
            success, message = execute_and_test(user_code, exercise.test_assertions)
            
            if success:
                console.print(Panel(message, title="✅ Success!", style="green"))
                progress = update_progress_on_success(progress)
                save_progress(progress)
                
                # Ask to continue
                console.print("\n")
                try:
                    choice = prompt("Press Enter for the next challenge, or type 'q' to quit: ")
                    if choice.strip().lower() == 'q':
                        return
                except (KeyboardInterrupt, EOFError):
                    return
                
                break # Break out of inner loop to get the next exercise
                
            else:
                console.print(Panel(message, title="❌ Failed", style="red"))
                progress = update_progress_on_failure(progress)
                save_progress(progress)
                
                # Generate a hint
                with console.status("[bold cyan]The sensei is pondering a hint...[/bold cyan]"):
                    hint = generate_hint(exercise, user_code, message)
                    
                console.print(Panel(hint, title="💡 Hint", style="magenta"))
                
                # Keep their code for the next iteration of the retry loop
                current_code = user_code
                console.print("\n[bold dim]Try again! Your previous code is loaded below.[/bold dim]")
            
    console.print("[bold green]Thanks for training at the Python Dojo![/bold green]")

if __name__ == "__main__":
    main_loop()

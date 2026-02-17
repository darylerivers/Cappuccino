#!/usr/bin/env python3
"""
GLM Code Assistant - Makes GLM-4 function like Claude Code
Provides file access, code analysis, and interactive assistance
"""

import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
import readline  # For command history


class GLMCodeAssistant:
    """
    Interactive coding assistant using GLM-4
    Simulates Claude Code behavior with file access and project context
    """

    def __init__(
        self,
        project_root: str = "/opt/user-data/experiment/cappuccino",
        model: str = "glm4",
        ollama_url: str = "http://localhost:11434"
    ):
        self.project_root = Path(project_root)
        self.model = model
        self.ollama_url = ollama_url
        self.conversation = []
        self.file_cache = {}  # Cache recently read files

        # Load system prompt
        self.system_prompt = self._create_system_prompt()

        print(f"GLM Code Assistant initialized")
        print(f"Project: {self.project_root}")
        print(f"Model: {model}")
        print()

    def _create_system_prompt(self) -> str:
        """Create Claude Code-like system prompt for GLM (optimized for speed)"""
        return f"""You are a coding assistant for the Cappuccino crypto trading system (Python, PyTorch, DRL agents).

Project: {self.project_root}

Your job: Help with code, debug issues, explain architecture.

Key files:
- scripts/training/1_optimize_unified.py (training)
- drl_agents/agents/AgentPPO_FT.py (agent)
- scripts/deployment/paper_trader_alpaca_polling.py (trading)

When you need files, say: "I need FILE_PATH"
Be concise. Reference line numbers. Provide examples."""

    def _call_glm(self, user_message: str, temperature: float = 0.3) -> str:
        """Call GLM-4 with conversation context"""

        # Build full context
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history (last 10 messages for context)
        for msg in self.conversation[-10:]:
            messages.append(msg)

        # Add current message
        messages.append({"role": "user", "content": user_message})

        # Format for GLM (concatenate with role labels)
        prompt = ""
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            prompt += f"{role}: {content}\n\n"

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 400  # Shorter = faster
                    }
                },
                timeout=60
            )

            result = response.json()
            return result.get('response', '').strip()

        except Exception as e:
            return f"Error calling GLM: {e}"

    def read_file(self, file_path: str) -> str:
        """Read a file and cache it"""
        try:
            full_path = self.project_root / file_path
            if not full_path.exists():
                return f"Error: File not found: {file_path}"

            # Check cache
            if file_path in self.file_cache:
                return self.file_cache[file_path]

            # Read file
            with open(full_path, 'r') as f:
                content = f.read()

            # Cache it
            self.file_cache[file_path] = content

            # Format with line numbers
            lines = content.split('\n')
            numbered = '\n'.join([f"{i+1:4d} | {line}" for i, line in enumerate(lines)])

            return f"File: {file_path}\n\n{numbered}"

        except Exception as e:
            return f"Error reading {file_path}: {e}"

    def run_command(self, command: str) -> str:
        """Run a shell command and return output"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"

            return output

        except Exception as e:
            return f"Error running command: {e}"

    def search_code(self, pattern: str) -> str:
        """Search for pattern in codebase"""
        try:
            result = subprocess.run(
                f"grep -r --include='*.py' '{pattern}' .",
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )

            return result.stdout if result.stdout else "No matches found"

        except Exception as e:
            return f"Error searching: {e}"

    def list_files(self, directory: str = ".") -> str:
        """List files in directory"""
        try:
            full_path = self.project_root / directory
            if not full_path.exists():
                return f"Error: Directory not found: {directory}"

            files = []
            for item in sorted(full_path.iterdir()):
                if item.is_file():
                    size = item.stat().st_size
                    files.append(f"  {item.name} ({size:,} bytes)")
                elif item.is_dir() and not item.name.startswith('.'):
                    files.append(f"  {item.name}/")

            return f"Directory: {directory}\n\n" + '\n'.join(files[:50])

        except Exception as e:
            return f"Error listing directory: {e}"

    def process_user_input(self, user_input: str) -> str:
        """
        Process user input and handle special commands
        Returns GLM's response
        """

        # Check for special commands
        if user_input.startswith('/read '):
            file_path = user_input[6:].strip()
            content = self.read_file(file_path)
            # Add to conversation
            self.conversation.append({
                "role": "user",
                "content": f"Here's the file you requested ({file_path}):\n\n{content}"
            })
            return f"Loaded file: {file_path}\n\n{content[:500]}..." + "\n\n(File loaded into context. Ask me about it!)"

        elif user_input.startswith('/run '):
            command = user_input[5:].strip()
            output = self.run_command(command)
            return f"Command output:\n\n{output}"

        elif user_input.startswith('/search '):
            pattern = user_input[8:].strip()
            results = self.search_code(pattern)
            return f"Search results for '{pattern}':\n\n{results}"

        elif user_input.startswith('/ls '):
            directory = user_input[4:].strip()
            listing = self.list_files(directory)
            return listing

        elif user_input == '/help':
            return """GLM Code Assistant Commands:

/read <file>      - Read a file and add to context
/run <command>    - Run a shell command
/search <pattern> - Search code for pattern
/ls <directory>   - List files in directory
/clear            - Clear conversation history
/help             - Show this help
/quit             - Exit assistant

Regular questions are sent directly to GLM-4.
GLM can ask you to run these commands if it needs information.
"""

        elif user_input == '/clear':
            self.conversation.clear()
            self.file_cache.clear()
            return "Conversation history cleared."

        elif user_input == '/quit':
            return None

        # Regular conversation - send to GLM
        else:
            # Add user message to history
            self.conversation.append({
                "role": "user",
                "content": user_input
            })

            # Get GLM response
            response = self._call_glm(user_input)

            # Check if GLM needs files
            if "I need to see" in response or "Can you show me" in response:
                # GLM is requesting information
                print("\n" + "="*70)
                print("GLM is requesting information. You can use:")
                print("  /read <file>   - to provide file contents")
                print("  /search <term> - to search the codebase")
                print("  /ls <dir>      - to list directory contents")
                print("="*70 + "\n")

            # Add GLM response to history
            self.conversation.append({
                "role": "assistant",
                "content": response
            })

            return response

    def chat_loop(self):
        """Interactive chat loop"""
        print("="*70)
        print("GLM CODE ASSISTANT - Interactive Mode")
        print("="*70)
        print()
        print("Type your questions or use /help for commands")
        print("Press Ctrl+C or type /quit to exit")
        print()
        print("Example questions:")
        print("  - How does the training script work?")
        print("  - Why is my paper trader not executing trades?")
        print("  - Can you explain the PPO agent architecture?")
        print("  - Help me debug this error: [paste error]")
        print()

        while True:
            try:
                # Get user input
                user_input = input("\n\033[1;36mYou:\033[0m ").strip()

                if not user_input:
                    continue

                # Process input
                response = self.process_user_input(user_input)

                if response is None:  # /quit command
                    print("\nGoodbye!")
                    break

                # Display response
                print(f"\n\033[1;32mGLM:\033[0m {response}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

    def quick_question(self, question: str) -> str:
        """Ask a single question without entering chat loop"""
        self.conversation.append({
            "role": "user",
            "content": question
        })

        response = self._call_glm(question)

        self.conversation.append({
            "role": "assistant",
            "content": response
        })

        return response


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GLM Code Assistant - Interactive coding help"
    )
    parser.add_argument(
        '--project-root',
        default='/opt/user-data/experiment/cappuccino',
        help='Project root directory'
    )
    parser.add_argument(
        '--question',
        '-q',
        help='Ask a single question (non-interactive)'
    )
    parser.add_argument(
        '--file',
        '-f',
        help='Load a file into context before asking question'
    )

    args = parser.parse_args()

    # Create assistant
    assistant = GLMCodeAssistant(project_root=args.project_root)

    # Non-interactive mode
    if args.question:
        # Load file if provided
        if args.file:
            content = assistant.read_file(args.file)
            assistant.conversation.append({
                "role": "system",
                "content": f"File loaded: {args.file}\n\n{content}"
            })

        # Ask question
        response = assistant.quick_question(args.question)
        print(response)

    # Interactive mode
    else:
        assistant.chat_loop()


if __name__ == "__main__":
    main()

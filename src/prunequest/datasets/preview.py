"""
Dataset & Prompt Preview Utilities for PruneQuest.
"""

from typing import List, Dict, Any, Optional
from rich.table import Table


class DatasetPreviewer:
    """
    Renders prompts, formats choices, and generates safe preview representations.
    """

    @classmethod
    def render_prompt(
        cls,
        input_text: str,
        template: str = "{{ input }}",
        system_prompt: Optional[str] = None,
        choices: Optional[List[str]] = None,
    ) -> str:
        prompt_body = template.replace("{{ input }}", input_text.strip())

        if choices and isinstance(choices, list):
            choice_lines = []
            for i, c in enumerate(choices):
                label = chr(65 + i) if i < 26 else str(i)
                choice_lines.append(f"{label}) {c}")
            prompt_body = f"{prompt_body}\n\n" + "\n".join(choice_lines)

        if system_prompt:
            return f"System: {system_prompt.strip()}\n\nUser: {prompt_body}"
        return prompt_body

    @classmethod
    def format_preview_item(
        cls,
        sample_id: str,
        input_text: str,
        target_text: str,
        task_type: str,
        scorer: str,
        template: str = "{{ input }}",
        system_prompt: Optional[str] = None,
        choices: Optional[List[str]] = None,
        max_chars: int = 120,
    ) -> Dict[str, Any]:
        rendered = cls.render_prompt(input_text, template, system_prompt, choices)

        def truncate(s: str) -> str:
            clean = s.replace("\n", " ")
            return clean[:max_chars] + ("..." if len(clean) > max_chars else "")

        return {
            "id": sample_id,
            "raw_input": truncate(input_text),
            "rendered_prompt": truncate(rendered),
            "target": truncate(target_text),
            "task_type": task_type,
            "scorer": scorer,
        }

    @classmethod
    def build_preview_table(cls, preview_items: List[Dict[str, Any]]) -> Table:
        table = Table(title="PruneQuest Dataset Prompt Preview", show_lines=True)
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Rendered Model Prompt", style="cyan")
        table.add_column("Expected Answer", style="green")
        table.add_column("Task / Scorer", style="magenta")

        for item in preview_items:
            table.add_row(
                str(item.get("id", "")),
                str(item.get("rendered_prompt", "")),
                str(item.get("target", "")),
                f"{item.get('task_type', '')}\n({item.get('scorer', '')})",
            )
        return table

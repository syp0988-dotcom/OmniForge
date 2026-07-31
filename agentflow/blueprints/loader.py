"""BlueprintLoader — load, match, and render Blueprints.

Architecture::

    BlueprintLoader
      ├── _load_all()             scan YAML files → list[Blueprint]
      ├── match(goal, goal_type)  keyword scoring → Top-N → LLM confirm → Blueprint | None
      ├── render(bp, config)      Jinja2 → list[FileSpec] with rendered content
      └── list()                  available blueprint IDs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import FileSystemLoader, TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment

from agentflow.blueprints.configurator import ProjectConfig
from agentflow.blueprints.models import Blueprint, FileSpec
from agentflow.utils.logging import build_logger

logger = build_logger("blueprint_loader")

# Default path: <package_root>/blueprints/
_DEFAULT_DIR = Path(__file__).resolve().parent


class BlueprintLoader:
    """Load, match, and render project blueprints."""

    def __init__(self, blueprints_dir: str | Path | None = None) -> None:
        self._dir = Path(blueprints_dir) if blueprints_dir else _DEFAULT_DIR
        self._templates_dir = self._dir / "templates"
        self._jinja = SandboxedEnvironment(
            loader=FileSystemLoader(str(self._templates_dir)),
            autoescape=False,
        )
        self._blueprints: list[Blueprint] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, goal: str, goal_type: str = "project") -> Blueprint | None:
        """Find the best matching blueprint for a goal.

        1. Fast keyword scoring → Top-3 candidates
        2. LLM confirmation → pick the best (or ``None`` if none fit)
        3. If LLM is unavailable → return Top-1

        Returns ``None`` when no blueprint matches (caller falls back to LLM).
        """
        self._ensure_loaded()

        # Only match project-type goals
        if goal_type != "project":
            logger.info("Blueprint: skipping match for goal_type=%s", goal_type)
            return None

        candidates = self._score(goal)
        if not candidates:
            logger.info("Blueprint: no candidates matched for goal")
            return None

        if len(candidates) == 1:
            logger.info("Blueprint: single candidate '%s' selected by score", candidates[0].id)
            return candidates[0]

        # Multi-candidate: try LLM confirmation
        try:
            return self._llm_confirm(goal, candidates) or candidates[0]
        except Exception as exc:
            logger.warning("Blueprint: LLM confirm failed (%s), using Top-1 '%s'", exc, candidates[0].id)
            return candidates[0]

    def render(
        self,
        blueprint: Blueprint,
        config: ProjectConfig,
        existing_files: set[str] | None = None,
    ) -> list[FileSpec]:
        """Render a blueprint's file templates with project variables.

        Args:
            blueprint: The blueprint to render.
            config: Project variables (from ProjectConfigurator).
            existing_files: Set of relative paths that already exist.
                           Files that exist are marked as type ``"skip"``.

        Returns:
            List of :class:`FileSpec` with rendered content.
        """
        existing = existing_files or set()
        variables = config.to_dict()
        specs: list[FileSpec] = []

        for file_spec in blueprint.files:
            # Check condition
            if file_spec.condition:
                condition_value = variables.get(file_spec.condition, False)
                if not condition_value:
                    logger.debug("Blueprint: skipping %s (condition '%s' is falsey)",
                                 file_spec.path, file_spec.condition)
                    continue

            # Resolve content
            rendered_path = self._render_path(file_spec.path, variables)
            rendered_content = self._resolve_content(file_spec, variables)

            # Determine action
            file_type = file_spec.type
            if rendered_path in existing:
                file_type = "skip"
                logger.info("Blueprint: '%s' exists — skipping", rendered_path)

            specs.append(FileSpec(
                path=rendered_path,
                content_template=rendered_content,
                type=file_type,
                description=file_spec.description,
                encoding=file_spec.encoding,
            ))

        return specs

    def list(self) -> list[str]:
        """Return all loaded blueprint IDs."""
        self._ensure_loaded()
        return [bp.id for bp in self._blueprints]

    def get(self, blueprint_id: str) -> Blueprint | None:
        """Get a blueprint by ID."""
        self._ensure_loaded()
        for bp in self._blueprints:
            if bp.id == blueprint_id:
                return bp
        return None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._blueprints = self._load_all()
            self._loaded = True

    def _load_all(self) -> list[Blueprint]:
        """Scan the blueprints directory and load all ``*.yaml`` files."""
        if not self._dir.exists() or not self._dir.is_dir():
            logger.warning("Blueprint directory not found: %s", self._dir)
            return []

        blueprints: list[Blueprint] = []
        for yaml_path in sorted(self._dir.glob("*.yaml")):
            # Skip __init__.py or other non-blueprint files
            if yaml_path.name.startswith("_"):
                continue
            try:
                bp = Blueprint.from_yaml(yaml_path)
                blueprints.append(bp)
                logger.info("Blueprint loaded: '%s' (%s) — %d files",
                            bp.id, bp.name, len(bp.files))
            except Exception as exc:
                logger.warning("Failed to load blueprint '%s': %s", yaml_path.name, exc)

        return blueprints

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _score(self, goal: str) -> list[Blueprint]:
        """Score all blueprints against the goal and return sorted by score.

        Scoring formula::

            score = keyword_hits * 0.3 + framework_hit * 0.4
                    - exclude_hit * 0.5
        """
        goal_lower = goal.lower()
        scored: list[tuple[float, Blueprint]] = []

        for bp in self._blueprints:
            score = 0.0

            # Keyword boost
            for kw in bp.keywords:
                if kw in goal_lower:
                    score += 0.3

            # Framework matches
            for fw in bp.frameworks:
                if fw in goal_lower:
                    score += 0.4

            # Exclusion penalties
            for ek in bp.exclude_keywords:
                if ek in goal_lower:
                    score -= 0.5

            if score > 0:
                scored.append((score, bp))

        scored.sort(key=lambda x: x[0], reverse=True)
        logger.debug("Blueprint scores: %s",
                     [(bp.id, round(s, 2)) for s, bp in scored])
        return [bp for _, bp in scored]

    def _llm_confirm(
        self,
        goal: str,
        candidates: list[Blueprint],
    ) -> Blueprint | None:
        """Use LLM to pick the best blueprint from candidates.

        The LLM can also reject all candidates (return ``None``),
        in which case the caller falls back to raw LLM generation.
        """
        from agentflow.services.llm_service import get_llm_service

        llm = get_llm_service()
        bp_descriptions = "\n\n".join(
            f"ID: {bp.id}\n名称: {bp.name}\n描述: {bp.description}\n框架: {', '.join(bp.frameworks)}"
            for bp in candidates
        )
        prompt = (
            f"用户目标：{goal}\n\n"
            f"候选蓝图：\n{bp_descriptions}\n\n"
            "请根据用户目标选择最合适的蓝图。如果都不合适，输出 null。"
            "只输出 JSON，格式：{\"blueprint_id\": \"...\"} 或 {\"blueprint_id\": null}"
        )

        import json
        raw = llm.complete(messages=[
            {"role": "system", "content": "你是一个项目架构师，为用户项目选择最合适的蓝图模板。只输出 JSON。"},
            {"role": "user", "content": prompt},
        ])
        raw = raw.strip()
        for marker in ("```json", "```JSON", "```"):
            if marker in raw:
                raw = raw.split(marker, 1)[-1]
                raw = raw.rsplit("```", 1)[0]
                raw = raw.strip()

        parsed: dict[str, Any] = json.loads(raw)
        selected_id = parsed.get("blueprint_id")
        if not selected_id:
            logger.info("Blueprint: LLM rejected all candidates")
            return None

        for bp in candidates:
            if bp.id == selected_id:
                logger.info("Blueprint: LLM selected '%s'", bp.id)
                return bp

        logger.warning("Blueprint: LLM selected '%s' but not in candidates", selected_id)
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _resolve_content(self, spec: FileSpec, variables: dict[str, Any]) -> str:
        """Resolve a FileSpec's content, preferring external templates."""
        if spec.template_ref:
            try:
                template = self._jinja.get_template(spec.template_ref)
                return template.render(**variables)
            except TemplateNotFound:
                logger.warning("Blueprint: external template '%s' not found, using inline",
                               spec.template_ref)
            except Exception as exc:
                logger.warning("Blueprint: template '%s' render failed (%s), using inline",
                               spec.template_ref, exc)

        # Fall back to inline template
        if spec.content_template:
            try:
                tpl = self._jinja.from_string(spec.content_template)
                return tpl.render(**variables)
            except Exception as exc:
                logger.warning("Blueprint: inline template render failed: %s", exc)
                return spec.content_template

        return ""

    def _render_path(self, path_template: str, variables: dict[str, Any]) -> str:
        """Render a file path template (may contain Jinja2 expressions)."""
        try:
            tpl = self._jinja.from_string(path_template)
            return tpl.render(**variables)
        except Exception as exc:
            logger.warning("Blueprint: path template render failed: %s", exc)
            return path_template

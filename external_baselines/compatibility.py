"""Portable vision-checkpoint configuration for the legacy Dispider loader."""
from __future__ import annotations

import ast
from pathlib import Path


class _PortableVisionLoader(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(isinstance(target, ast.Attribute) and target.attr == "proxies" for target in node.targets):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        node.keywords = [keyword for keyword in node.keywords if keyword.arg != "proxies"]
        return node


def configure_dispider_source(install_dir: Path) -> None:
    """Use user networking and an explicit vision checkpoint in the native loader."""
    path = install_dir / "dispider/model/multimodal_encoder/clip_encoder.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree = _PortableVisionLoader().visit(tree)
    if not any(isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names) for node in tree.body):
        tree.body.insert(0, ast.Import(names=[ast.alias(name="os")]))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CLIPVisionTower")
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    if not any(isinstance(node, ast.Constant) and node.value == "DSB_VISION_CHECKPOINT" for node in ast.walk(constructor)):
        override = ast.parse('vision_tower = os.environ.get("DSB_VISION_CHECKPOINT", vision_tower)').body[0]
        constructor.body.insert(0, override)
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")


def verify_dispider_source(install_dir: Path) -> None:
    path = install_dir / "dispider/model/multimodal_encoder/clip_encoder.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "proxies":
            raise ValueError("Run the Dispider source setup before inference")
        if isinstance(node, ast.keyword) and node.arg == "proxies":
            raise ValueError("Run the Dispider source setup before inference")
    if not any(isinstance(node, ast.Constant) and node.value == "DSB_VISION_CHECKPOINT" for node in ast.walk(tree)):
        raise ValueError("Run the Dispider source setup before inference")

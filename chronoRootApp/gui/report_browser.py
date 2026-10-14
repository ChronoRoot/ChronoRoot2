"""Build a filesystem mirror catalog of report figures for the Tab5 viewer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Union

from analysis.utils.report_utils import natural_key


@dataclass
class ReportLeaf:
    label: str
    plot_file: str
    stats_file: str = ''
    description: str = ''


@dataclass
class ReportBranch:
    label: str
    children: List = field(default_factory=list)

    def sorted_children(self):
        self.children.sort(key=lambda node: natural_key(node.label))
        for child in self.children:
            if isinstance(child, ReportBranch):
                child.sorted_children()


def _insert_png(root: Dict, rel_plot: str, rel_stats: str):
    parts = rel_plot.replace('\\', '/').split('/')
    if len(parts) == 1:
        root[parts[0]] = ReportLeaf(label=parts[0], plot_file=rel_plot, stats_file=rel_stats)
        return

    node = root
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = ReportLeaf(label=parts[-1], plot_file=rel_plot, stats_file=rel_stats)


def _dict_to_nodes(tree: Dict) -> List[Union[ReportBranch, ReportLeaf]]:
    nodes: List[Union[ReportBranch, ReportLeaf]] = []
    for label, child in tree.items():
        if isinstance(child, ReportLeaf):
            nodes.append(child)
        elif isinstance(child, dict):
            branch = ReportBranch(label=label, children=_dict_to_nodes(child))
            if branch.children:
                nodes.append(branch)
    nodes.sort(key=lambda node: natural_key(node.label))
    return nodes


def visible_in_report_preview(plot_file: str) -> bool:
    """Whether the main Report tab shows this figure.

    Per-plant growth curves, plant-count figures, and angle overlays stay
    on disk. The tab is a preview of the report folder.
    """
    rel = (plot_file or '').replace('\\', '/')
    if not rel:
        return False
    parts = rel.split('/')
    name = parts[-1]
    if rel.startswith('individual_plots/'):
        return False
    if any(part.startswith('overlays_') for part in parts[:-1]):
        return False
    if name == 'all_metrics_n.png' or name.endswith('_count.png'):
        return False
    return True


def filter_report_preview(nodes: List[Union[ReportBranch, ReportLeaf]]):
    """Drop preview-hidden figures and folders that then have nothing to show."""
    kept = []
    for node in nodes:
        if isinstance(node, ReportBranch):
            children = filter_report_preview(node.children)
            if not children:
                continue
            node.children = children
            kept.append(node)
        elif visible_in_report_preview(getattr(node, 'plot_file', '')):
            kept.append(node)
    return kept


def load_report_catalog(report_root: str) -> List[Union[ReportBranch, ReportLeaf]]:
    """Build tree from disk: one branch per folder segment, one leaf per .png."""
    if not os.path.isdir(report_root):
        return []

    tree: Dict = {}
    for dirpath, _, filenames in os.walk(report_root):
        for filename in filenames:
            if not filename.lower().endswith('.png'):
                continue
            abs_plot = os.path.join(dirpath, filename)
            rel_plot = os.path.relpath(abs_plot, report_root).replace('\\', '/')
            stats_abs = os.path.join(dirpath, f'{os.path.splitext(filename)[0]}_stats.txt')
            rel_stats = ''
            if os.path.isfile(stats_abs):
                rel_stats = os.path.relpath(stats_abs, report_root).replace('\\', '/')
            _insert_png(tree, rel_plot, rel_stats)

    catalog = _dict_to_nodes(tree)
    for node in catalog:
        if isinstance(node, ReportBranch):
            node.sorted_children()
    return catalog

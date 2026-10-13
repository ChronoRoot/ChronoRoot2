"""
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import gzip
import xml.etree.ElementTree as ET
import networkx as nx

GRAPHS_FILENAME = "graphs.xml.gz"

def combined_graph_path(graphs_dir):
    return os.path.join(graphs_dir, GRAPHS_FILENAME)


def _local_tag(tag):
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _node_token(value):
    number = float(value)
    if number.is_integer():
        text = str(int(number))
    else:
        text = f"{number:.6f}".rstrip("0").rstrip(".")
    if text.startswith("-"):
        return "m" + text[1:]
    return text


def _node_id(pos, used):
    base = f"n{_node_token(pos[0])}_{_node_token(pos[1])}"
    node_id = base
    suffix = 2
    while node_id in used:
        node_id = f"{base}_{suffix}"
        suffix += 1
    used.add(node_id)
    return node_id


def _data(parent, key, value):
    element = ET.SubElement(parent, "data", {"key": key})
    element.text = value
    return element


def _graph_xml(graph, frame_idx, image_name):
    """One GraphML graph element. Does not modify the live graph."""
    element = ET.Element("graph", {
        "id": f"g{int(frame_idx)}",
        "edgedefault": "undirected",
    })
    _data(element, "g_frame", str(int(frame_idx)))
    _data(element, "g_image", os.path.basename(image_name or ""))

    used = set()
    ids = {}
    for node, data in graph.nodes(data=True):
        pos = data["pos"]
        node_id = _node_id(pos, used)
        ids[node] = node_id
        node_el = ET.SubElement(element, "node", {"id": node_id})
        _data(node_el, "n_type", str(data.get("type", "")))
        _data(node_el, "n_age", str(int(data.get("age", 0))))
        _data(node_el, "n_x", repr(float(pos[0])))
        _data(node_el, "n_y", repr(float(pos[1])))

    for u, v, data in graph.edges(data=True):
        edge_el = ET.SubElement(element, "edge", {"source": ids[u], "target": ids[v]})
        _data(edge_el, "e_weight", repr(float(data.get("weight", 0))))
        _data(edge_el, "e_color", str(int(data.get("color", 0))))
        _data(edge_el, "e_root", str(int(data.get("root_type", 0))))
        if data.get("traversal_cost") is not None:
            _data(edge_el, "e_cost", repr(float(data["traversal_cost"])))
    return ET.tostring(element, encoding="unicode") + "\n"


_GRAPHML_HEADER = """<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="g_frame" for="graph" attr.name="frame_index" attr.type="long"/>
  <key id="g_image" for="graph" attr.name="image_name" attr.type="string"/>
  <key id="n_type" for="node" attr.name="type" attr.type="string"/>
  <key id="n_age" for="node" attr.name="age" attr.type="long"/>
  <key id="n_x" for="node" attr.name="pos_x" attr.type="double"/>
  <key id="n_y" for="node" attr.name="pos_y" attr.type="double"/>
  <key id="e_weight" for="edge" attr.name="weight" attr.type="float"/>
  <key id="e_color" for="edge" attr.name="color" attr.type="long"/>
  <key id="e_root" for="edge" attr.name="root_type" attr.type="long"/>
  <key id="e_cost" for="edge" attr.name="traversal_cost" attr.type="float"/>
"""


class GraphSequenceWriter:
    """One compressed GraphML document; each tracked frame is its own graph."""

    def __init__(self, conf):
        self.path = combined_graph_path(conf["folders"]["graphs"])
        self._closed = False
        self._handle = gzip.open(self.path, "wt", encoding="utf-8")
        try:
            self._handle.write(_GRAPHML_HEADER)
            self._handle.flush()
        except Exception:
            self._handle.close()
            self._closed = True
            raise

    def add(self, graph, frame_idx, image_name):
        if self._closed or graph is False or graph is None:
            if graph is False or graph is None:
                print(f"Not valid graph for {image_name}")
            return
        self._handle.write(_graph_xml(graph, frame_idx, image_name))
        self._handle.flush()

    def close(self):
        if self._closed:
            return
        self._closed = True
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.write("</graphml>\n")
        finally:
            handle.close()


def saveGraph(graph, conf, image_name):
    """
    Save graph structure to disk as compressed GraphML (XML format).
    
    Args:
        graph: NetworkX graph with node attributes (pos, type, age) 
               and edge attributes (weight, color, root_type)
               Can also be False/None for failed frames
        conf: Configuration dictionary with output folders
        image_name: Name of the image file (will replace .png with .xml.gz)
    """
    save_filename = image_name.replace('.png', '.xml.gz')
    save_path = os.path.join(conf['folders']['graphs'], save_filename)
    
    if graph is not False and graph is not None:
        # Convert tuple positions to strings for GraphML compatibility
        # GraphML doesn't natively support tuple attributes
        graph_copy = graph.copy()
        
        for node in graph_copy.nodes():
            pos = graph_copy.nodes[node]['pos']
            # Store position as "x,y" string
            graph_copy.nodes[node]['pos_x'] = float(pos[0])
            graph_copy.nodes[node]['pos_y'] = float(pos[1])
            # Remove tuple attribute (GraphML can't serialize tuples)
            del graph_copy.nodes[node]['pos']
        
        # Write compressed GraphML
        with gzip.open(save_path, 'wb') as f:
            # Convert to bytes for gzip
            graphml_string = '\n'.join(nx.generate_graphml(graph_copy))
            f.write(graphml_string.encode('utf-8'))
    else:
        print(f'Not valid graph for {image_name}')
    
    return


def saveProps(image_name, frame_number, graph, csv_writer, number_lateral_roots, hypocotyl_length):
    """
    Extract and save measurements from the graph to CSV.
    
    Args:
        image_name: Name of the image file
        frame_number: Frame index in sequence
        graph: NetworkX graph (or False/None for failed frames)
        csv_writer: CSV writer object
        number_lateral_roots: Count of lateral roots from RSML creation
    """
    if graph is not False and graph is not None:
        # Calculate lengths by summing edge weights
        main_root_length = 0
        lateral_roots_length = 0
        total_length = 0
        
        for u, v, data in graph.edges(data=True):
            edge_weight = data.get('weight', 0)
            edge_type = data.get('root_type', 0)
            
            total_length += edge_weight
            
            if edge_type == 10:
                # Main root edge
                main_root_length += edge_weight
            else:
                # Lateral root edge
                lateral_roots_length += edge_weight
        
        row = [
            image_name, 
            frame_number, 
            main_root_length, 
            lateral_roots_length, 
            number_lateral_roots, 
            total_length,
            hypocotyl_length
        ]
    else:
        # No valid graph - write zeros
        row = [image_name, frame_number, 0, 0, 0, 0, 0]
    
    csv_writer.writerow(row)
    return

def _restore_positions(graph):
    for node in graph.nodes():
        pos_x = float(graph.nodes[node].get("pos_x", 0))
        pos_y = float(graph.nodes[node].get("pos_y", 0))
        graph.nodes[node]["pos"] = (pos_x, pos_y)
    return graph


def _cast_graphml(text, typ):
    if text is None:
        return None
    if typ in ("long", "int", "integer"):
        return int(float(text))
    if typ in ("double", "float"):
        return float(text)
    if typ == "boolean":
        return str(text).strip().lower() in ("1", "true", "yes")
    return text


def _apply_graphml_data(target, elements, keys, domain):
    for data in elements:
        if _local_tag(data.tag) != "data":
            continue
        spec = keys.get(data.get("key"))
        if not spec or spec[0] != domain:
            continue
        target[spec[1]] = _cast_graphml(data.text, spec[2])


def _graph_from_element(graph_el, keys):
    graph = nx.Graph()
    graph_data = {}
    _apply_graphml_data(graph_data, list(graph_el), keys, "graph")
    graph.graph.update(graph_data)
    for child in graph_el:
        kind = _local_tag(child.tag)
        if kind == "node":
            attrs = {}
            _apply_graphml_data(attrs, list(child), keys, "node")
            graph.add_node(child.get("id"), **attrs)
        elif kind == "edge":
            attrs = {}
            _apply_graphml_data(attrs, list(child), keys, "edge")
            graph.add_edge(child.get("source"), child.get("target"), **attrs)
    return _restore_positions(graph)


def _same_image(stored, wanted):
    if not stored or not wanted:
        return False
    left = os.path.basename(str(stored))
    right = os.path.basename(str(wanted))
    if left == right:
        return True

    def stem(name):
        lower = name.lower()
        for ext in (".png", ".tif", ".tiff", ".rsml"):
            if lower.endswith(ext):
                return name[:-len(ext)]
        return os.path.splitext(name)[0]

    return stem(left) == stem(right)


def _graph_matches(graph_el, keys, image_name, frame_index):
    data = {}
    _apply_graphml_data(data, list(graph_el), keys, "graph")
    if frame_index is not None:
        stored = data.get("frame_index")
        id_match = (graph_el.get("id") or "") == f"g{int(frame_index)}"
        if not (id_match or (stored is not None and int(stored) == int(frame_index))):
            return False
    if image_name is not None and not _same_image(data.get("image_name"), image_name):
        return False
    return True


def _open_graphml(filepath):
    if str(filepath).endswith(".gz"):
        return gzip.open(filepath, "rb")
    return open(filepath, "rb")


def _load_combined_graph(filepath, image_name=None, frame_index=None, which=None):
    keys = {}
    chosen = None
    with _open_graphml(filepath) as handle:
        for _event, elem in ET.iterparse(handle, events=("end",)):
            kind = _local_tag(elem.tag)
            if kind == "key":
                keys[elem.get("id")] = (
                    elem.get("for"),
                    elem.get("attr.name"),
                    elem.get("attr.type"),
                )
                elem.clear()
                continue
            if kind != "graph":
                continue
            take = which == "last" or _graph_matches(elem, keys, image_name, frame_index)
            if which is None and image_name is None and frame_index is None:
                take = chosen is None
            if take:
                chosen = _graph_from_element(elem, keys)
                if which != "last" and (image_name is not None or frame_index is not None or which is None):
                    elem.clear()
                    break
            elem.clear()
    if chosen is None:
        raise KeyError(
            f"Graph not found in {filepath} "
            f"(image_name={image_name}, frame_index={frame_index}, which={which})"
        )
    return chosen


def _load_legacy_graph(filepath):
    with _open_graphml(filepath) as handle:
        graph = nx.read_graphml(handle)
    return _restore_positions(graph)


def loadGraph(filepath, image_name=None, frame_index=None, which=None):
    """
    Load one graph from a combined graphs.xml.gz or a legacy single-graph file.

    Combined files are indexed by graph id `g{frame_index}` and by image_name.
    `which="last"` returns the last graph in the file. A legacy file is the
    one graph it contains; NetworkX reads that file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    if os.path.basename(filepath) == GRAPHS_FILENAME:
        return _load_combined_graph(
            filepath,
            image_name=image_name,
            frame_index=frame_index,
            which=which,
        )
    return _load_legacy_graph(filepath)

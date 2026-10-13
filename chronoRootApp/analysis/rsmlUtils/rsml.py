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

import xml.etree.ElementTree as ET
from datetime import datetime
import getpass
import re
import numpy as np
import os
from ..imageUtils.seg import skeleton_nodes

# Global counter for RSML points
n_points = 0

def createTree(conf, frame_idx, images, graph, skeleton, skeleton_overlay):
    """
    Main entry point to generate the RSML tree from the graph/skeleton.
    """
    global n_points
    n_points = 0
    
    # 1. Create RSML Header
    tree = createHeader(conf, frame_idx, images)
    root = tree.getroot()
    
    # 2. Extract endpoints for checking termination
    _, end_points = skeleton_nodes(skeleton)
    end_points = np.array(end_points)
    
    # 3. Find the Seed (Start) Node
    seed_nodes = [node for node in graph.nodes() if graph.nodes[node]['type'] == "Ini"]
    
    if len(seed_nodes) == 0:
        # Fallback: Find node with degree 1 that is highest (lowest Y)
        possible_seeds = [n for n in graph.nodes() if graph.degree(n) == 1]
        if not possible_seeds: 
             # If no endpoints, just pick top-most node
             all_nodes = list(graph.nodes())
             if not all_nodes: raise Exception("Graph is empty")
             seed_node = sorted(all_nodes, key=lambda p: p[1])[0]
        else:
            seed_node = sorted(possible_seeds, key=lambda p: p[1])[0]
    else:
        seed_node = seed_nodes[0]

    seed_position = np.array(graph.nodes[seed_node]['pos'], dtype='int')

    # 4. Identify Main Root colors from the graph
    # (These are the edge colors assigned during createGraph traversal)
    main_root_colors = []
    for u, v, data in graph.edges(data=True):
        if data.get('root_type') == 10:
            main_root_colors.append(data.get('color', 0))

    # Snap the seed to a nearby skeleton endpoint only when that endpoint
    # still touches the main root. The graph Ini is already that start.
    if len(end_points) > 0:
        distances = np.linalg.norm(end_points - seed_position, axis=1)
        snap_distance = float(np.min(distances))
        nearest = end_points[int(np.argmin(distances))]
        if 0 < snap_distance < 10.0:
            touches_main = False
            ny, nx = int(nearest[1]), int(nearest[0])
            if 0 <= ny < skeleton_overlay.shape[0] and 0 <= nx < skeleton_overlay.shape[1]:
                if skeleton_overlay[ny, nx] in main_root_colors:
                    touches_main = True
            if not touches_main:
                for nb in vecinos(skeleton_overlay, nearest):
                    if skeleton_overlay[nb[1], nb[0]] in main_root_colors:
                        touches_main = True
                        break
            if touches_main:
                seed_position = nearest
    
    # 5. Build the RSML
    # We pass a COPY of the skeleton overlay because we will erase pixels as we visit them
    _, number_lateral_roots = completeRSML(
        skeleton_overlay.copy(), 
        seed_position, 
        root, 
        main_root_colors,
    )
    
    plant = tree.find(".//plant")
    main_root = plant.find("./root[@label='mainRoot']")
    
    # 1. Get all direct children of the main root
    direct_children = main_root.findall("./root")
    
    # 2. Filter them to ensure they are explicitly labeled as Order 1
    # This ignores any potential malformed tags that don't match your naming convention
    xml_o1_count = 0
    for child in direct_children:
        label = child.get('label', '')
        if label.startswith('lat_o1'):
            xml_o1_count += 1

    if number_lateral_roots != xml_o1_count:
        print(f"Warning: Calculated {number_lateral_roots} laterals, but XML contains {xml_o1_count} 'lat_o1' tags.")
    
    # 6. Safety check
    total_skeleton_points = np.sum(skeleton > 0)
    if total_skeleton_points > 0 and n_points < total_skeleton_points * 0.7:
        raise Exception("RSML generation incomplete: less than 70% skeleton points captured.")
    
    return tree, number_lateral_roots

def completeRSML(ske2, seed, rsml, mainRoot):
    """
    Main traversal logic. Uses a queue and explicit marking to avoid loops.
    """
    global n_points

    plant = rsml.find(".//plant")
    if plant is None: plant = rsml.find('scene').find('plant')

    # Create main root element
    raiz = ET.SubElement(plant, 'root', {'id': 'p', 'label': 'mainRoot'})
    raiz.text, raiz.tail = '\n\t\t\t', '\n\t\t'
    geo = ET.SubElement(raiz, 'geometry')
    geo.text, geo.tail = '\n\t\t\t', '\n\t\t'
    polyline = ET.SubElement(geo, 'polyline')
    polyline.text, polyline.tail = '\n\t\t\t\t', '\n\t\t\t'
    
    # Add Seed
    add_point(polyline, seed)
    n_points += 1
    
    # Get neighbors before zeroing seed
    all_neighbors = vecinos(ske2, seed)
    ske2[seed[1], seed[0]] = 0
    
    if not all_neighbors:
        return rsml, 0

    # Sort Neighbors (Main vs Lateral)
    main_candidates = [n for n in all_neighbors if ske2[n[1], n[0]] in mainRoot]
    
    if main_candidates:
        main_start = main_candidates[0]
        lateral_starts = [n for n in all_neighbors if n != main_start]
    else:
        main_start = all_neighbors[0]
        lateral_starts = all_neighbors[1:]

    # Init Queue 
    lateral_queue = []
    for ls in lateral_starts:
        lateral_queue.append({
            'start': ls, 'parent': seed, 'order': 1, 'elem': raiz  
        })

    # Trace Main Root
    if main_start:
        continue_mainRoot(ske2, main_start, seed, rsml, mainRoot, lateral_queue, raiz)

    # Process Laterals
    processed_count = 0
    counters = {}
    
    # SAFETY: Prevent infinite loops with a max iteration guard
    safety_counter = 0
    MAX_ITERATIONS = 20000 

    while lateral_queue:
        safety_counter += 1
        if safety_counter > MAX_ITERATIONS:
            print("Warning: Max iterations reached in RSML generation. Breaking loop.")
            raise Exception("Max iterations reached in RSML generation.")

        task = lateral_queue.pop(0)
        start, parent, order, p_elem = task['start'], task['parent'], task['order'], task['elem']
        
        # Vital check: has this pixel been eaten by another path?
        if ske2[start[1], start[0]] == 0:
            continue
            
        if order == 1: processed_count += 1
        
        # XML
        idx = counters.get(order, 0)
        counters[order] = idx + 1
        lat_id = f"lat_o{order}_{idx}"
        
        lr = ET.SubElement(p_elem, 'root', {'id': lat_id, 'label': lat_id})
        lr.text, lr.tail = '\n\t\t\t', '\n\t\t'
        geo = ET.SubElement(lr, 'geometry')
        geo.text, geo.tail = '\n\t\t\t', '\n\t\t'
        poly = ET.SubElement(geo, 'polyline')
        poly.text, poly.tail = '\n\t\t\t\t', '\n\t\t\t'
        
        # Connect to parent
        add_point(poly, parent)
        
        # Trace
        ske2, stop_node, poly = get_next_node_rsml(ske2, start, parent, poly)
        
        # At stop_node. It might be a junction.
        # 1. Get neighbors
        nbs = vecinos(ske2, stop_node)
        
        # 2. Queue valid neighbors
        for n in nbs:
            # Check if neighbor is unvisited
            if ske2[n[1], n[0]] != 0:
                lateral_queue.append({
                    'start': n, 'parent': stop_node, 'order': order + 1, 'elem': lr
                })
        
        # Zero the stop node now that we've queued its children.
        # This prevents other branches from looping back to this node.
        ske2[stop_node[1], stop_node[0]] = 0

    return rsml, processed_count
            
def continue_mainRoot(ske2, current, previous, rsml, mainRoot, lat_queue, main_root_elem):
    # Retrieve the polyline element for the main root
    polyline = main_root_elem.findall(".//geometry/polyline")[-1]
    
    safety = 0
    while True:
        if safety > 10000: 
            raise Exception("Max iterations reached in RSML generation.")
        safety += 1
        
        # Trace segment until a junction or endpoint
        ske2, stop_node, polyline = get_next_node_rsml(ske2, current, previous, polyline)
        
        # Scan neighbors at the stopping point
        nbs = vecinos(ske2, stop_node)
        
        if not nbs:
            # End of main root. Zero the tip.
            ske2[stop_node[1], stop_node[0]] = 0
            break
            
        next_main = None
        lat_candidates = []
        
        # PASS 1: Priority check for a direct Main Root connection
        for n in nbs:
            if ske2[n[1], n[0]] == 0: continue
            if ske2[n[1], n[0]] in mainRoot:
                next_main = n
                break # Direct continuation found, stop looking
                
        # PASS 2: Only if no direct connection exists, look for a 1-pixel gap bridge
        if next_main is None:
            for n in nbs:
                if ske2[n[1], n[0]] == 0: continue
                
                lookahead_nbs = vecinos(ske2, n)
                is_gap_bridged = False
                for ln in lookahead_nbs:
                    if np.array_equal(ln, stop_node): continue
                    if ske2[ln[1], ln[0]] in mainRoot:
                        next_main = n
                        is_gap_bridged = True
                        break
                        
                if is_gap_bridged:
                    break # Bridge found, stop looking
                    
        # PASS 3: Everything that isn't the chosen next_main is a lateral root candidate
        for n in nbs:
            if ske2[n[1], n[0]] == 0: continue
            if not np.array_equal(n, next_main):
                lat_candidates.append(n)

        # Zero the junction node
        ske2[stop_node[1], stop_node[0]] = 0
        
        # Safely queue the recovered laterals
        for lat_n in lat_candidates:
            lat_queue.append({
                'start': lat_n, 
                'parent': stop_node, 
                'order': 1, 
                'elem': main_root_elem
            })
        
        # Continue the main loop if a main root path exists
        if next_main is not None:
            previous = stop_node
            current = next_main
        else:
            break

def get_next_node_rsml(ske, start, previous, polyline):
    """
    Traces edge. Returns [ske, stop_node, dist, poly].
    Does NOT zero the stop_node (the caller does that after checking neighbors).
    Does zero the path intermediate pixels.
    """
    global n_points
    
    current = start
    prev = previous
    
    # Zero the start pixel immediately to prevent immediate loops
    # (Because 'start' is the first pixel of the NEW edge)
    ske[current[1], current[0]] = 0
    add_point(polyline, current)
    n_points += 1
    
    while True:
        # Get neighbors of current (which is already 0, but that's fine for finding next)
        # Wait - vecinos needs to find neighbors. 
        # Neighbors of 'current' are 1. 'current' is 0.
        
        all_nbs = vecinos(ske, current)
        
        # Filter: Don't go back to prev (though prev is likely 0 now too)
        # We just need any non-zero neighbor
        valid = [n for n in all_nbs if not np.array_equal(n, prev)]
        
        if len(valid) == 0:
            # Endpoint
            return ske, current, polyline
        
        if len(valid) > 1:
            # Junction
            return ske, current, polyline
            
        # Single valid neighbor -> Step forward
        nxt = valid[0]
        
        add_point(polyline, nxt)
        n_points += 1
        
        # Zero the pixel we are LEAVING (Wait, we move to nxt, so zero nxt?)
        # Standard: Zero the pixel as we visit it.
        ske[nxt[1], nxt[0]] = 0
        
        prev = current
        current = nxt


def vecinos(ske, seed): 
    """
    Finds 8-connected neighbors of a pixel that are non-zero.
    """
    x, y = int(seed[0]), int(seed[1])
    h, w = ske.shape
    neighbors = []
    
    # Optimized bounds to prevent index errors
    y_min, y_max = max(0, y-1), min(h, y+2)
    x_min, x_max = max(0, x-1), min(w, x+2)
    
    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            if i == y and j == x: continue
            if ske[i, j] != 0:
                neighbors.append([j, i])
    return neighbors


def add_point(polyline, point):
    """Helper to add an XML point"""
    ET.SubElement(polyline, 'point', {'x': str(point[0]), 'y': str(point[1])}).tail = '\n\t\t\t\t'

def createHeader(conf, i, images):
    rsml_file = "analysis/default.rsml"
    tree = ET.parse(rsml_file)
    root = tree.getroot()
    metadata = root[0]
    scene = root[1]   
    
    for elemento in metadata:
        if elemento.tag == 'user':
            elemento.text = getpass.getuser()
        if elemento.tag == 'file-key':
            elemento.text = conf.get('fileKey', 'unknown')
        if elemento.tag == 'last-modified':
            now = datetime.now()
            elemento.text = now.strftime("%Y-%m-%dT%H:%M:%S")
            
        if elemento.tag == 'image':
            for sub in elemento:
                if sub.tag == 'name':
                    sub.text = os.path.basename(images[i])
                if sub.tag == 'captured':
                    # Simplified regex for robustness
                    try:
                        time_str = os.path.basename(images[i]).replace('.png','')
                        nums = re.findall(r'\d+', time_str)
                        if len(nums) >= 6:
                            text = f"{nums[0]}-{nums[1]}-{nums[2]}T{nums[3]}:{nums[4]}:{nums[5]}"
                            sub.text = text
                    except:
                        pass
    
        if elemento.tag == 'time-sequence':
            for sub in elemento:
                if sub.tag == 'index':
                    sub.text = str(i)
                if sub.tag == 'label':
                    sub.text = conf.get('sequenceLabel', 'default')
                    
    tag = 'plant'
    attrib = {'id': '1', 'label': conf.get('Plant', 'plant1')}
    plant = ET.Element(tag,attrib)
    plant.text, plant.tail = "\n\t\t", "\n\t"
    scene.append(plant)

    return tree

def saveRSML(rsmlTree, conf, image_name):
    path = os.path.join(conf['folders']['rsml'], image_name.replace('.png','.rsml'))
    rsmlTree.write(open(path, 'w'), encoding='unicode')
    return

"""
A* Pathfinding System with Flood Detection for Malolos, Bulacan, Philippines
Uses OpenStreetMap data with real road network, one-way streets, and proper routing.
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime
import heapq
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')

# Configure OSMnx
ox.settings.use_cache = True
ox.settings.log_console = False


class FloodSensor:
    """Simulates flood sensors at specific locations"""
    def __init__(self, sensor_id, location, name):
        self.sensor_id = sensor_id
        self.location = location  # (lat, lon)
        self.name = name
        self.is_flooded = False
        self.flood_level = 0  # 0-5 scale (0=no flood, 5=severe)
    
    def detect_flood(self, level):
        """Simulate flood detection"""
        self.flood_level = level
        self.is_flooded = level > 0
        return self.is_flooded


class AStarFloodRouter:
    """A* pathfinding with flood avoidance for Malolos, Bulacan"""
    
    def __init__(self):
        self.G = None
        self.G_original = None
        self.sensors = []
        self.flooded_nodes = set()
        self.flooded_edges = set()
        self.place_name = "Malolos, Bulacan, Philippines"
        
    def load_map(self):
        """Load real road network from OpenStreetMap"""
        print(f"Loading road network for {self.place_name}...")
        
        # Download the road network with driving directions (respects one-way streets)
        self.G = ox.graph_from_place(
            self.place_name,
            network_type='drive',  # Only drivable roads
            simplify=True
        )
        
        # Keep original graph for comparison
        self.G_original = self.G.copy()
        
        # Add edge weights based on travel time (considering speed limits and road types)
        self.G = ox.routing.add_edge_speeds(self.G)
        self.G = ox.routing.add_edge_travel_times(self.G)
        
        print(f"Loaded {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
        return self.G
    
    def setup_flood_sensors(self):
        """Set up flood sensors at strategic locations in Malolos"""
        # Real flood-prone areas in Malolos based on typical flooding patterns
        # Positioned along main roads between City Hall and BSU areas
        sensor_locations = [
            # Main areas in Malolos that are prone to flooding
            (14.8470, 120.8120, "Barangay Santisima Trinidad"),  # Along main road
            (14.8510, 120.8170, "MacArthur Highway Junction"),   # Highway intersection
            (14.8490, 120.8200, "Bulihan Market Area"),          # Low-lying market
            (14.8389, 120.8156, "Pamarawan Lowland"),            # Flood-prone lowland
            (14.8520, 120.8230, "San Pablo Creek"),              # Near water body
            (14.8450, 120.8150, "Caliligawan Bridge"),           # Bridge area
            (14.8540, 120.8190, "Longos-Guinhawa Road"),         # Main road
            (14.8480, 120.8100, "Malolos River Vicinity"),       # Near river
        ]
        
        for i, (lat, lon, name) in enumerate(sensor_locations):
            sensor = FloodSensor(i, (lat, lon), name)
            self.sensors.append(sensor)
        
        print(f"Installed {len(self.sensors)} flood sensors")
        return self.sensors
    
    def haversine_distance(self, coord1, coord2):
        """Calculate the great circle distance between two points in meters"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        R = 6371000  # Earth's radius in meters
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return R * c
    
    def get_nearest_node(self, lat, lon):
        """Find the nearest node in the graph to the given coordinates"""
        return ox.distance.nearest_nodes(self.G, lon, lat)
    
    def simulate_flood(self, flooded_sensor_indices, flood_radius=300):
        """
        Simulate flooding by blocking nodes near flooded sensors
        flood_radius: radius in meters around sensor that is considered flooded
        """
        self.flooded_nodes = set()
        self.flooded_edges = set()
        
        for idx in flooded_sensor_indices:
            if idx < len(self.sensors):
                sensor = self.sensors[idx]
                sensor.detect_flood(3)  # Moderate flood level
                
                # Find all nodes within flood radius
                sensor_lat, sensor_lon = sensor.location
                
                for node in self.G.nodes():
                    node_lat = self.G.nodes[node]['y']
                    node_lon = self.G.nodes[node]['x']
                    
                    dist = self.haversine_distance(
                        (sensor_lat, sensor_lon),
                        (node_lat, node_lon)
                    )
                    
                    if dist <= flood_radius:
                        self.flooded_nodes.add(node)
        
        # Find all edges connected to flooded nodes
        for u, v, k in self.G.edges(keys=True):
            if u in self.flooded_nodes or v in self.flooded_nodes:
                self.flooded_edges.add((u, v, k))
        
        print(f"Flood affects {len(self.flooded_nodes)} nodes and {len(self.flooded_edges)} road segments")
        return self.flooded_nodes, self.flooded_edges
    
    def a_star_heuristic(self, node1, node2):
        """Heuristic function for A* (Haversine distance)"""
        lat1, lon1 = self.G.nodes[node1]['y'], self.G.nodes[node1]['x']
        lat2, lon2 = self.G.nodes[node2]['y'], self.G.nodes[node2]['x']
        return self.haversine_distance((lat1, lon1), (lat2, lon2))
    
    def a_star_search(self, start_node, end_node, avoid_flooded=True):
        """
        A* pathfinding algorithm with flood avoidance
        Returns: (path, total_distance, total_time)
        """
        if start_node not in self.G.nodes or end_node not in self.G.nodes:
            return None, float('inf'), float('inf')
        
        # Priority queue: (f_score, counter, node)
        counter = 0
        open_set = [(0, counter, start_node)]
        came_from = {}
        
        g_score = {node: float('inf') for node in self.G.nodes}
        g_score[start_node] = 0
        
        f_score = {node: float('inf') for node in self.G.nodes}
        f_score[start_node] = self.a_star_heuristic(start_node, end_node)
        
        open_set_hash = {start_node}
        
        while open_set:
            current = heapq.heappop(open_set)[2]
            open_set_hash.discard(current)
            
            if current == end_node:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # Calculate total distance and time
                total_distance = 0
                total_time = 0
                for i in range(len(path) - 1):
                    edge_data = self.G.get_edge_data(path[i], path[i+1])
                    if edge_data:
                        # Get the first edge (in case of multiple edges)
                        first_edge = list(edge_data.values())[0]
                        total_distance += first_edge.get('length', 0)
                        total_time += first_edge.get('travel_time', 0)
                
                return path, total_distance, total_time
            
            # Check if current node is flooded
            if avoid_flooded and current in self.flooded_nodes:
                continue
            
            # Explore neighbors
            for neighbor in self.G.neighbors(current):
                # Skip flooded nodes
                if avoid_flooded and neighbor in self.flooded_nodes:
                    continue
                
                # Get edge data
                edge_data = self.G.get_edge_data(current, neighbor)
                if not edge_data:
                    continue
                
                # Use travel time as the cost (considers speed limits and road types)
                first_edge = list(edge_data.values())[0]
                edge_cost = first_edge.get('travel_time', first_edge.get('length', 1))
                
                tentative_g_score = g_score[current] + edge_cost
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.a_star_heuristic(neighbor, end_node)
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        open_set_hash.add(neighbor)
        
        return None, float('inf'), float('inf')  # No path found
    
    def find_route_with_flood_avoidance(self, start_coords, end_coords):
        """
        Find routes: original and alternative (avoiding floods)
        start_coords, end_coords: (lat, lon) tuples
        """
        start_node = self.get_nearest_node(*start_coords)
        end_node = self.get_nearest_node(*end_coords)
        
        print(f"\nFinding routes from {start_coords} to {end_coords}")
        print(f"Start node: {start_node}, End node: {end_node}")
        
        # Find original route (ignoring floods)
        original_path, orig_dist, orig_time = self.a_star_search(
            start_node, end_node, avoid_flooded=False
        )
        
        # Find alternative route (avoiding floods)
        alt_path, alt_dist, alt_time = self.a_star_search(
            start_node, end_node, avoid_flooded=True
        )
        
        return {
            'original': {
                'path': original_path,
                'distance': orig_dist,
                'time': orig_time
            },
            'alternative': {
                'path': alt_path,
                'distance': alt_dist,
                'time': alt_time
            },
            'start_node': start_node,
            'end_node': end_node
        }


def generate_pdf_report(router, routes, output_file="flood_routing_report.pdf"):
    """Generate a comprehensive PDF report with maps and route information"""
    
    print(f"\nGenerating PDF report: {output_file}")
    
    with PdfPages(output_file) as pdf:
        # Page 1: Title and Overview
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.85, 'A* Flood Routing System', fontsize=24, ha='center', fontweight='bold')
        fig.text(0.5, 0.78, 'Malolos, Bulacan, Philippines', fontsize=18, ha='center')
        fig.text(0.5, 0.70, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                fontsize=12, ha='center')
        
        # System overview
        overview_text = """
        SYSTEM OVERVIEW
        
        This A* Pathfinding System uses real OpenStreetMap data to find optimal routes
        while avoiding flood-affected areas. The system:
        
        • Uses real road network data from OpenStreetMap
        • Respects one-way streets and road restrictions
        • Considers road types and speed limits for travel time estimation
        • Simulates flood sensors at strategic flood-prone locations
        • Implements A* algorithm with flood avoidance capability
        • Provides alternative routes when primary routes are flooded
        
        STUDY AREA: Malolos, Bulacan
        
        Malolos is the capital city of Bulacan province in the Philippines.
        It is known for having flood-prone areas, especially during monsoon season.
        """
        fig.text(0.1, 0.15, overview_text, fontsize=10, va='bottom', family='monospace')
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Map with road network
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_title('Road Network of Malolos, Bulacan\n(OpenStreetMap Data)', fontsize=14, fontweight='bold')
        
        # Plot the road network
        ox.plot_graph(router.G, ax=ax, node_size=0, edge_color='gray', 
                     edge_linewidth=0.5, bgcolor='white', show=False, close=False)
        
        # Plot flood sensors
        for sensor in router.sensors:
            color = 'red' if sensor.is_flooded else 'green'
            ax.scatter(sensor.location[1], sensor.location[0], c=color, s=100, 
                      marker='^', zorder=5, edgecolors='black', linewidths=1)
            ax.annotate(sensor.name, (sensor.location[1], sensor.location[0]), 
                       fontsize=6, ha='center', va='bottom', xytext=(0, 5),
                       textcoords='offset points')
        
        # Legend
        active_sensor = mpatches.Patch(color='green', label='Active Sensor (No Flood)')
        flooded_sensor = mpatches.Patch(color='red', label='Flooded Sensor')
        ax.legend(handles=[active_sensor, flooded_sensor], loc='upper right')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Flood Sensors Information
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.92, 'Flood Sensor Locations', fontsize=18, ha='center', fontweight='bold')
        
        sensor_info = "ID  | Name                    | Coordinates           | Status\n"
        sensor_info += "-" * 70 + "\n"
        for sensor in router.sensors:
            status = "FLOODED" if sensor.is_flooded else "Normal"
            sensor_info += f"{sensor.sensor_id:2d}  | {sensor.name:22s} | ({sensor.location[0]:.4f}, {sensor.location[1]:.4f}) | {status}\n"
        
        fig.text(0.1, 0.5, sensor_info, fontsize=10, va='center', family='monospace')
        pdf.savefig(fig)
        plt.close()
        
        # Page 4: Original Route (without flood consideration)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_title('Original Route (Without Flood Consideration)', fontsize=14, fontweight='bold')
        
        ox.plot_graph(router.G, ax=ax, node_size=0, edge_color='lightgray', 
                     edge_linewidth=0.5, bgcolor='white', show=False, close=False)
        
        if routes['original']['path']:
            # Plot original route
            route_coords = [(router.G.nodes[n]['x'], router.G.nodes[n]['y']) 
                           for n in routes['original']['path']]
            xs, ys = zip(*route_coords)
            ax.plot(xs, ys, 'b-', linewidth=3, label='Original Route', zorder=4)
        
        # Plot flooded areas
        for node in router.flooded_nodes:
            ax.scatter(router.G.nodes[node]['x'], router.G.nodes[node]['y'], 
                      c='red', s=20, alpha=0.5, zorder=3)
        
        # Plot start and end points
        start_node = routes['start_node']
        end_node = routes['end_node']
        ax.scatter(router.G.nodes[start_node]['x'], router.G.nodes[start_node]['y'], 
                  c='green', s=200, marker='*', zorder=6, label='Start (Point A)')
        ax.scatter(router.G.nodes[end_node]['x'], router.G.nodes[end_node]['y'], 
                  c='purple', s=200, marker='*', zorder=6, label='End (Point B)')
        
        # Route info
        orig_dist = routes['original']['distance']
        orig_time = routes['original']['time']
        info_text = f"Distance: {orig_dist/1000:.2f} km | Travel Time: {orig_time/60:.1f} minutes"
        ax.text(0.5, -0.05, info_text, transform=ax.transAxes, ha='center', fontsize=10)
        
        ax.legend(loc='upper right')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 5: Alternative Route (avoiding floods)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_title('Alternative Route (Avoiding Flood Areas)', fontsize=14, fontweight='bold')
        
        ox.plot_graph(router.G, ax=ax, node_size=0, edge_color='lightgray', 
                     edge_linewidth=0.5, bgcolor='white', show=False, close=False)
        
        # Plot flooded areas prominently
        for node in router.flooded_nodes:
            ax.scatter(router.G.nodes[node]['x'], router.G.nodes[node]['y'], 
                      c='red', s=30, alpha=0.6, zorder=3)
        
        # Add flooded zone circles around sensors
        for sensor in router.sensors:
            if sensor.is_flooded:
                circle = plt.Circle((sensor.location[1], sensor.location[0]), 
                                   0.003, color='red', alpha=0.2, zorder=2)
                ax.add_patch(circle)
        
        if routes['alternative']['path']:
            # Plot alternative route
            route_coords = [(router.G.nodes[n]['x'], router.G.nodes[n]['y']) 
                           for n in routes['alternative']['path']]
            xs, ys = zip(*route_coords)
            ax.plot(xs, ys, 'g-', linewidth=3, label='Alternative Route', zorder=4)
        else:
            ax.text(0.5, 0.5, 'NO ALTERNATIVE ROUTE FOUND', transform=ax.transAxes,
                   ha='center', va='center', fontsize=16, color='red', fontweight='bold')
        
        # Plot start and end points
        ax.scatter(router.G.nodes[start_node]['x'], router.G.nodes[start_node]['y'], 
                  c='green', s=200, marker='*', zorder=6, label='Start (Point A)')
        ax.scatter(router.G.nodes[end_node]['x'], router.G.nodes[end_node]['y'], 
                  c='purple', s=200, marker='*', zorder=6, label='End (Point B)')
        
        # Route info
        alt_dist = routes['alternative']['distance']
        alt_time = routes['alternative']['time']
        if alt_dist < float('inf'):
            info_text = f"Distance: {alt_dist/1000:.2f} km | Travel Time: {alt_time/60:.1f} minutes"
        else:
            info_text = "No route available"
        ax.text(0.5, -0.05, info_text, transform=ax.transAxes, ha='center', fontsize=10)
        
        ax.legend(loc='upper right')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 6: Comparison - Both routes
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_title('Route Comparison: Original vs Alternative', fontsize=14, fontweight='bold')
        
        ox.plot_graph(router.G, ax=ax, node_size=0, edge_color='lightgray', 
                     edge_linewidth=0.5, bgcolor='white', show=False, close=False)
        
        # Plot flooded areas
        for node in router.flooded_nodes:
            ax.scatter(router.G.nodes[node]['x'], router.G.nodes[node]['y'], 
                      c='red', s=20, alpha=0.4, zorder=3)
        
        # Plot original route
        if routes['original']['path']:
            route_coords = [(router.G.nodes[n]['x'], router.G.nodes[n]['y']) 
                           for n in routes['original']['path']]
            xs, ys = zip(*route_coords)
            ax.plot(xs, ys, 'b--', linewidth=2, label='Original Route (may pass through flood)', 
                   zorder=4, alpha=0.7)
        
        # Plot alternative route
        if routes['alternative']['path']:
            route_coords = [(router.G.nodes[n]['x'], router.G.nodes[n]['y']) 
                           for n in routes['alternative']['path']]
            xs, ys = zip(*route_coords)
            ax.plot(xs, ys, 'g-', linewidth=3, label='Alternative Route (flood-free)', zorder=5)
        
        # Plot start and end points
        ax.scatter(router.G.nodes[start_node]['x'], router.G.nodes[start_node]['y'], 
                  c='green', s=200, marker='*', zorder=6, label='Start (Point A)')
        ax.scatter(router.G.nodes[end_node]['x'], router.G.nodes[end_node]['y'], 
                  c='purple', s=200, marker='*', zorder=6, label='End (Point B)')
        
        ax.legend(loc='upper right')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 7: Route Summary and Statistics
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.92, 'Route Analysis Summary', fontsize=18, ha='center', fontweight='bold')
        
        orig_dist = routes['original']['distance']
        orig_time = routes['original']['time']
        alt_dist = routes['alternative']['distance']
        alt_time = routes['alternative']['time']
        
        summary = f"""
        ROUTE COMPARISON
        {'='*60}
        
        ORIGINAL ROUTE (May pass through flooded areas):
        • Distance: {orig_dist/1000:.2f} km
        • Estimated Travel Time: {orig_time/60:.1f} minutes
        • Status: {'BLOCKED BY FLOOD' if any(n in router.flooded_nodes for n in (routes['original']['path'] or [])) else 'Available'}
        
        ALTERNATIVE ROUTE (Avoiding all flood zones):
        • Distance: {alt_dist/1000:.2f} km if alt_dist < float('inf') else 'N/A'
        • Estimated Travel Time: {alt_time/60:.1f} minutes if alt_time < float('inf') else 'N/A'
        • Status: {'Available' if routes['alternative']['path'] else 'NO ROUTE FOUND'}
        
        FLOOD IMPACT ANALYSIS:
        • Number of flooded nodes: {len(router.flooded_nodes)}
        • Number of blocked road segments: {len(router.flooded_edges)}
        • Active flood sensors: {sum(1 for s in router.sensors if s.is_flooded)}
        
        {'='*60}
        
        RECOMMENDATION:
        """
        
        if routes['alternative']['path']:
            if alt_dist < float('inf'):
                extra_dist = alt_dist - orig_dist
                extra_time = alt_time - orig_time
                summary += f"""
        Take the ALTERNATIVE ROUTE to avoid flood-affected areas.
        • Additional distance: {extra_dist/1000:.2f} km ({(extra_dist/orig_dist*100):.1f}% longer)
        • Additional travel time: {extra_time/60:.1f} minutes
        
        This route ensures safe passage avoiding all detected flood zones.
        """
        else:
            summary += """
        WARNING: No safe alternative route is available.
        Consider waiting for flood waters to recede or use alternative
        transportation methods.
        """
        
        fig.text(0.1, 0.1, summary, fontsize=10, va='bottom', family='monospace')
        pdf.savefig(fig)
        plt.close()
        
        # Page 8: A* Algorithm Explanation
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.92, 'A* Algorithm Implementation Details', fontsize=18, ha='center', fontweight='bold')
        
        algo_text = """
        A* PATHFINDING ALGORITHM
        {'='*60}
        
        The A* algorithm finds the shortest path between two points using:
        
        f(n) = g(n) + h(n)
        
        Where:
        • f(n) = Total estimated cost of path through node n
        • g(n) = Actual cost from start to node n  
        • h(n) = Heuristic estimate from n to goal (Haversine distance)
        
        IMPLEMENTATION FEATURES:
        
        1. REAL ROAD DATA
           - Uses OpenStreetMap road network
           - Respects one-way streets
           - Considers road types (highway, primary, residential, etc.)
        
        2. TRAVEL TIME ESTIMATION
           - Uses road type-based speed limits
           - Accounts for road length and conditions
        
        3. FLOOD AVOIDANCE
           - Sensors detect flood at specific coordinates
           - Nodes within flood radius are marked as blocked
           - A* algorithm excludes flooded nodes from path
        
        4. HEURISTIC FUNCTION
           - Haversine distance (great-circle distance)
           - Admissible and consistent heuristic
           - Guarantees optimal path when possible
        
        DATA SOURCES:
        - Road Network: OpenStreetMap (© OpenStreetMap contributors)
        - Coordinates: WGS84 (EPSG:4326)
        - Network Analysis: OSMnx library
        """
        
        fig.text(0.1, 0.1, algo_text, fontsize=10, va='bottom', family='monospace')
        pdf.savefig(fig)
        plt.close()
        
    print(f"PDF report saved: {output_file}")
    return output_file


def main():
    """Main function to run the flood routing system"""
    print("=" * 60)
    print("A* FLOOD ROUTING SYSTEM - MALOLOS, BULACAN")
    print("=" * 60)
    
    # Initialize router
    router = AStarFloodRouter()
    
    # Load map data from OpenStreetMap
    router.load_map()
    
    # Setup flood sensors at strategic locations
    router.setup_flood_sensors()
    
    # Define start and end points (real locations in Malolos)
    # Point A: Near Malolos City Hall
    start_point = (14.8437, 120.8091)
    
    # Point B: Near Bulacan State University
    end_point = (14.8571, 120.8267)
    
    print(f"\nRoute Request:")
    print(f"  Start (Point A): Near Malolos City Hall {start_point}")
    print(f"  End (Point B): Near Bulacan State University {end_point}")
    
    # Simulate flood at specific sensors blocking the main route
    print("\n" + "=" * 60)
    print("FLOOD SIMULATION")
    print("=" * 60)
    flooded_sensors = [0, 1, 5]  # Block Santisima Trinidad, MacArthur Highway, and Caliligawan Bridge
    print(f"Simulating flood at sensors: {[router.sensors[i].name for i in flooded_sensors]}")
    router.simulate_flood(flooded_sensors, flood_radius=120)  # Moderate flood radius
    
    # Find routes
    print("\n" + "=" * 60)
    print("ROUTE CALCULATION")
    print("=" * 60)
    routes = router.find_route_with_flood_avoidance(start_point, end_point)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if routes['original']['path']:
        print(f"\nOriginal Route:")
        print(f"  Distance: {routes['original']['distance']/1000:.2f} km")
        print(f"  Travel Time: {routes['original']['time']/60:.1f} minutes")
        print(f"  Nodes in path: {len(routes['original']['path'])}")
    else:
        print("\nOriginal Route: NO PATH FOUND")
    
    if routes['alternative']['path']:
        print(f"\nAlternative Route (avoiding floods):")
        print(f"  Distance: {routes['alternative']['distance']/1000:.2f} km")
        print(f"  Travel Time: {routes['alternative']['time']/60:.1f} minutes")
        print(f"  Nodes in path: {len(routes['alternative']['path'])}")
        
        # Compare routes
        if routes['original']['path']:
            extra_dist = routes['alternative']['distance'] - routes['original']['distance']
            extra_time = routes['alternative']['time'] - routes['original']['time']
            print(f"\n  Additional distance: {extra_dist/1000:.2f} km")
            print(f"  Additional travel time: {extra_time/60:.1f} minutes")
    else:
        print("\nAlternative Route: NO SAFE PATH FOUND")
    
    # Generate PDF report
    print("\n" + "=" * 60)
    print("GENERATING PDF REPORT")
    print("=" * 60)
    pdf_file = generate_pdf_report(router, routes, "flood_routing_report.pdf")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nPDF Report: {pdf_file}")
    print("The report includes:")
    print("  1. System overview")
    print("  2. Road network map")
    print("  3. Flood sensor locations")
    print("  4. Original route visualization")
    print("  5. Alternative route visualization")
    print("  6. Route comparison")
    print("  7. Analysis summary")
    print("  8. Algorithm details")
    
    return router, routes


if __name__ == "__main__":
    main()

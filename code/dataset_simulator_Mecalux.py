import pandas as pd
import numpy as np
import random
import torch
from config import device
from scipy.stats import betabinom
import json

from nodes import Location, Coordinates, InventoryProduct, DemandNode, InventoryNode, InventoryNodeManager

node_locations = {
    'IT 20098' : (45.4642, 9.1900),  # Milan, Italy
    'PL 44-109' : (50.0647, 19.9450),  # Katowice, Poland
    'BR 02033-000' : (-15.8267, -47.9218),  # Brasília, Brazil
    'ES 33211' : (28.2916, -16.6291),  # Santa Cruz de Tenerife, Spain
    'ES 46210' : (39.4699, -0.3763),  # Valencia, Spain
    'AR C1651' : (-34.6037, -58.3816),  # Buenos Aires, Argentina
    'US 92154' : (32.5435, -117.0301),  # San Diego, United States
    'US 61764' : (40.1106, -88.2073),  # Rantoul, United States
    'IL 60160' : (41.9806, -88.0706),  # Melrose Park, United States
    'US 76207' : (33.2148, -97.1331),  # Denton, United States
    'DE 41540' : (51.1657, 7.1940),  # Dormagen, Germany
    'ES 28914' : (40.3384, -3.8134),  # Leganés, Spain
    'US 29151' : (33.6976, -80.2170),  # Sumter, United States
    'MX 45615' : (20.5888, -100.3899),  # San Juan del Río, Mexico
    'GB B6 7JJ' : (52.4862, -1.8904),  # Birmingham, United Kingdom
    'ES 34192' : (40.9634, -4.1571),  # Segovia, Spain
    'ES 89400' : (41.6488, -0.8891),  # Zaragoza, Spain
    'PL 80-299' : (54.3776, 18.4662),  # Gdańsk, Poland
    'TAM 87316' : (22.2560, -97.8349),  # Tampico, Mexico
    'ES 8940' : (41.6488, -0.8891),  # Zaragoza, Spain
    'MX 54030' : (19.4326, -99.1332),  # Mexico City, Mexico
    'MX 22444' : (32.5149, -117.0382),  # Tijuana, Mexico
    'FR 69740' : (45.7772, 4.8554),  # Genas, France
    'FR 91320' : (48.6975, 2.3522),  # Palaiseau, France
}

region_locations = {
    "LU": (49.6117, 6.1319),  # Luxembourg City, Luxembourg
    "RS": (44.8176, 20.4569),  # Belgrade, Serbia
    "GR": (37.9838, 23.7275),  # Athens, Greece
    "KW": (29.3759, 47.9774),  # Kuwait City, Kuwait
    "SG": (1.3521, 103.8198),  # Singapore, Singapore
    "RE": (-20.8789, 55.4481),  # Saint-Denis, Réunion
    "DO": (18.4861, -69.9312),  # Santo Domingo, Dominican Republic
    "GT": (14.6349, -90.5069),  # Guatemala City, Guatemala
    "SR": (5.8686, -55.1690),  # Paramaribo, Suriname
    "AD": (42.5077, 1.5211),  # Andorra la Vella, Andorra
    "US": (38.9072, -77.0369),  # Washington, D.C., United States
    "VE": (10.4806, -66.9036),  # Caracas, Venezuela
    "AE": (24.4539, 54.3773),  # Abu Dhabi, United Arab Emirates
    "FR": (48.8566, 2.3522),  # Paris, France
    "HN": (14.0723, -87.1921),  # Tegucigalpa, Honduras
    "IT": (41.9028, 12.4964),  # Rome, Italy
    "GY": (6.8013, -58.1551),  # Georgetown, Guyana
    "GB": (51.5074, -0.1278),  # London, United Kingdom
    "EC": (-0.1807, -78.4678),  # Quito, Ecuador
    "PL": (52.2297, 21.0122),  # Warsaw, Poland
    "NI": (12.1364, -86.2514),  # Managua, Nicaragua
    "CA": (45.4215, -75.6972),  # Ottawa, Canada
    "SC": (-4.6796, 55.4910),  # Victoria, Seychelles
    "SI": (46.0569, 14.5058),  # Ljubljana, Slovenia
    "TT": (10.6918, -61.2225),  # Port of Spain, Trinidad and Tobago
    "SN": (14.7167, -17.4677),  # Dakar, Senegal
    "BG": (42.6977, 23.3219),  # Sofia, Bulgaria
    "CR": (9.9281, -84.0907),  # San José, Costa Rica
    "BE": (50.8503, 4.3517),  # Brussels, Belgium
    "BD": (23.8103, 90.4125),  # Dhaka, Bangladesh
    "IL": (31.7683, 35.2137),  # Jerusalem, Israel
    "DE": (52.5200, 13.4050),  # Berlin, Germany
    "GI": (36.1408, -5.3536),  # Gibraltar, Gibraltar
    "MQ": (14.6415, -61.0242),  # Fort-de-France, Martinique
    "IE": (53.3498, -6.2603),  # Dublin, Ireland
    "GP": (16.2650, -61.5510),  # Basse-Terre, Guadeloupe
    "ET": (9.1450, 40.4897),  # Addis Ababa, Ethiopia
    "CZ": (50.0755, 14.4378),  # Prague, Czech Republic
    "HR": (45.8150, 15.9819),  # Zagreb, Croatia
    "MA": (34.0209, -6.8416),  # Rabat, Morocco
    "RO": (44.4268, 26.1025),  # Bucharest, Romania
    "TN": (36.8065, 10.1815),  # Tunis, Tunisia
    "BL": (17.9000, -62.8333),  # Gustavia, Saint Barthélemy
    "NC": (-22.2711, 166.4391),  # Nouméa, New Caledonia
    "CO": (4.7110, -74.0721),  # Bogotá, Colombia
    "KY": (19.3133, -81.2546),  # George Town, Cayman Islands
    "MX": (19.4326, -99.1332),  # Mexico City, Mexico
    "SM": (43.9336, 12.4507),  # San Marino, San Marino
    "QA": (25.2760, 51.2175),  # Doha, Qatar
    "KZ": (51.1694, 71.4491),  # Nur-Sultan (Astana), Kazakhstan
    "SA": (24.7136, 46.6753),  # Riyadh, Saudi Arabia
    "HU": (47.4979, 19.0402),  # Budapest, Hungary
    "AR": (-34.6037, -58.3816),  # Buenos Aires, Argentina
    "CY": (35.1856, 33.3823),  # Nicosia, Cyprus
    "EE": (59.4370, 24.7536),  # Tallinn, Estonia
    "IN": (28.6139, 77.2090),  # New Delhi, India
    "MK": (41.9981, 21.4254),  # Skopje, North Macedonia
    "ES": (40.4168, -3.7038),  # Madrid, Spain
    "SV": (13.6929, -89.2182),  # San Salvador, El Salvador
    "CL": (-33.4489, -70.6693),  # Santiago, Chile
    "TR": (39.9334, 32.8597),  # Ankara, Turkey
    "BR": (-15.8267, -47.9218),  # Brasília, Brazil
    "BA": (43.8563, 18.4131),  # Sarajevo, Bosnia and Herzegovina
    "CV": (14.9315, -23.5125),  # Praia, Cape Verde
    "AT": (48.2082, 16.3738),  # Vienna, Austria
    "MT": (35.8997, 14.5146),  # Valletta, Malta
    "CH": (46.9481, 7.4474),  # Bern, Switzerland
    "MC": (43.7384, 7.4246),  # Monaco, Monaco
    "PE": (-12.0464, -77.0428),  # Lima, Peru
    "SE": (59.3293, 18.0686),  # Stockholm, Sweden
    "LV": (56.9496, 24.1052),  # Riga, Latvia
    "SK": (48.1486, 17.1077),  # Bratislava, Slovakia
    "FI": (60.1699, 24.9384),  # Helsinki, Finland
    "NL": (52.3676, 4.9041),  # Amsterdam, Netherlands
    "DK": (55.6761, 12.5683),  # Copenhagen, Denmark
    "BO": (-16.4897, -68.1193),  # La Paz, Bolivia
    "PT": (38.7169, -9.1390),  # Lisbon, Portugal
    "NO": (59.9139, 10.7522),  # Oslo, Norway
    "IS": (64.1355, -21.8954),  # Reykjavik, Iceland
    "UY": (-34.9011, -56.1645),  # Montevideo, Uruguay
    "KR": (37.5665, 126.9780),  # Seoul, South Korea
    "PF": (-17.6797, -149.4068),  # Papeete, French Polynesia
    "AU": (-35.2809, 149.1300),  # Canberra, Australia
    "LT": (54.6872, 25.2797),  # Vilnius, Lithuania
}


class DatasetSimulator:
    def __init__(self, node_file, customer_file, sales_order_file, order_line_file, inventory_file):
        """Initialize the simulator with node, customer, sales order, and order line data."""
        # Load data
        self.node_data = pd.read_csv(node_file)
        self.customer_data = pd.read_csv(customer_file)
        self.sales_orders = pd.read_csv(sales_order_file)
        self.order_lines = pd.read_csv(order_line_file)
        self.inventory_data = pd.read_csv(inventory_file)

        # Sort sales orders by release date for sequential processing
        self.sales_orders["releasedate"] = pd.to_datetime(self.sales_orders["releasedate"])
        self.sales_orders = self.sales_orders.sort_values("releasedate").reset_index(drop=True)

        # Initialize nodes and customers
        self.inventory_nodes = self._initialize_nodes()
        self.customer_regions = self._initialize_customers()

        # Current sales order index (for simulation)
        self.current_order_idx = 0

        # Compute coordinate bounds and max distance
        self._coord_bounds = self._calculate_coord_bounds()
        self._max_dist = self._calculate_max_dist()

    def _initialize_nodes(self):
        """Initialize inventory nodes with location and inventory data."""
        inventory_nodes = []
        for _, node_row in self.node_data.iterrows():
            # Map ZIP code to coordinates
            if node_row["zipcode"] in node_locations:
                lat, lon = node_locations[node_row["zipcode"]]
            else:
                raise ValueError(f"ZIP code {node_row['zipcode']} not found in node_locations.")

            # Create a Location object for the node
            loc = Location(Coordinates(lat, lon))

            # Filter inventory data for this node
            node_inventory = self.inventory_data[self.inventory_data["nodeid"] == node_row["id"]]

            # Create InventoryProduct objects for the node's inventory
            inv_prods = [
                InventoryProduct(sku_id=inventory_row["itemid"], quantity=inventory_row["quantity"])
                for _, inventory_row in node_inventory.iterrows()
            ]

            # Initialize the InventoryNode
            inventory_node = InventoryNode(
                inv_prods=inv_prods,  # Correct argument name
                loc=loc,
                inv_node_id=node_row["id"],
                num_skus=len(node_inventory)
            )
            inventory_nodes.append(inventory_node)

        return inventory_nodes

    def _initialize_customers(self):
        """Initialize customer regions with location data."""
        customer_regions = {}
        for _, row in self.customer_data.iterrows():
            # Map region code to coordinates
            if row["code"] in region_locations:
                lat, lon = region_locations[row["code"]]
                customer_regions[row["id"]] = Location(Coordinates(lat, lon))
            else:
                raise ValueError(f"Region code {row['code']} not found in region_locations.")

        return customer_regions

    def gen_next_demand_node(self):
        """Generate the next demand node based on the next sales order."""
        if self.current_order_idx >= len(self.sales_orders):
            raise StopIteration("No more sales orders to process.")

        # Get the next sales order
        order = self.sales_orders.iloc[self.current_order_idx]
        self.current_order_idx += 1

        # Map destination ZIP code to location
        if order["destinationzipcode"] in region_locations:
            location = Location(Coordinates(*region_locations[order["destinationzipcode"]]))
        else:
            raise ValueError(f"Destination ZIP code {order['destinationzipcode']} not found in region_locations.")

        # Get the corresponding order lines
        order_lines = self.order_lines[self.order_lines["salesorderid"] == order["id"]]

        # Create InventoryProduct objects for the demand
        inv_products = [
            InventoryProduct(sku_id=row["itemid"], quantity=row["quantity"])
            for _, row in order_lines.iterrows()
        ]

        # Return the DemandNode
        return DemandNode(inv_products, location, len(region_locations))
    
    def get_inventory_nodes(self):
        """Return the initialized inventory nodes."""
        return self.inventory_nodes

    def reset(self):
        """Reset the simulator for a new simulation run."""
        self.current_order_idx = 0

    @property
    def num_skus(self):
        """Return the total number of unique SKUs in the dataset."""
        # Combine SKUs from inventory and order lines
        inventory_skus = set(self.inventory_data["itemid"].unique())
        order_line_skus = set(self.order_lines["itemid"].unique())

        # Return the total unique SKUs
        return len(inventory_skus.union(order_line_skus))
    
    def _calculate_coord_bounds(self):
        """Calculate the coordinate bounds based on node and customer locations."""
        node_coords = [
            (lat, lon) for _, (lat, lon) in node_locations.items()
        ]
        customer_coords = [
            (loc.coords.x, loc.coords.y) for loc in self.customer_regions.values()
        ]

        all_coords = node_coords + customer_coords

        min_lat = min(x[0] for x in all_coords)
        max_lat = max(x[0] for x in all_coords)
        min_lon = min(x[1] for x in all_coords)
        max_lon = max(x[1] for x in all_coords)

        return {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
        }

    def _calculate_max_dist(self):
        """Calculate the maximum distance between the farthest points."""
        # Get bounds from `_coord_bounds`
        c_1 = (self._coord_bounds["min_lat"], self._coord_bounds["min_lon"])
        c_2 = (self._coord_bounds["max_lat"], self._coord_bounds["max_lon"])

        # Calculate Euclidean distance
        return ((c_1[0] - c_2[0]) ** 2 + (c_1[1] - c_2[1]) ** 2) ** 0.5

    @property
    def coord_bounds(self):
        """Expose coordinate bounds."""
        return self._coord_bounds

    @property
    def max_dist(self):
        """Expose the maximum distance."""
        return self._max_dist
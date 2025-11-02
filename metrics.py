# metrics.py
import torch
import math

def haversine_distance(coord1, coord2):
    """
    Compute the haversine distance between two points.
    coord1, coord2: Tensors of shape (..., 2) in degrees [lat, lon]
    Returns distance in kilometers.
    """
    R = 6371.0  # Earth radius in km
    lat1, lon1 = torch.deg2rad(coord1[..., 0]), torch.deg2rad(coord1[..., 1])
    lat2, lon2 = torch.deg2rad(coord2[..., 0]), torch.deg2rad(coord2[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c
    return distance

def compute_metrics(pred_coords, true_coords, thresholds=[1, 25, 200, 750, 2500]):
    """
    pred_coords, true_coords: Tensors of shape (N, 2) in degrees [lat, lon]
    Returns a dictionary of percentages (in %) of predictions within specified thresholds.
    """
    distances = haversine_distance(pred_coords, true_coords)
    print("median distance", distances.median())
    metrics = {}
    for thresh in thresholds:
        metrics[f"within_{thresh}km"] = (distances <= thresh).float().mean().item() * 100
    return metrics



# metrics.py
import torch
from geopy.distance import geodesic

def compute_metrics_geodesic(pred_coords, true_coords, thresholds=[1,25,200,750,2500]):
    """
    pred_coords, true_coords: Tensors of shape (N, 2) in degrees [lat, lon]
    Uses geopy.geodesic under the hood (much slower than vectorized haversine).
    """
    # Convert to Python lists of tuples
    pred_list = [tuple(x.tolist()) for x in pred_coords]
    true_list = [tuple(x.tolist()) for x in true_coords]

    # Compute distances one by one
    dists = [geodesic(p, t).km for p, t in zip(pred_list, true_list)]
    distances = torch.tensor(dists)  # back into a tensor if you like

    # Build metrics
    metrics = {}
    for thresh in thresholds:
        metrics[f"within_{thresh}km"] = (distances <= thresh).float().mean().item() * 100
    return metrics


# metrics.py
import torch
from pyproj import Geod

# create a Geod once
_geod = Geod(ellps="WGS84")

def geodesic_distance_vectorized(pred_coords, true_coords):
    """
    pred_coords, true_coords: Tensors of shape (N,2) in degrees [lat,lon]
    Returns a Tensor of shape (N,) with distances in kilometers,
    computed on the WGS84 ellipsoid via a C‑vectorized PROJ call.
    """
    # pull out lon/lat as numpy arrays
    lats1 = pred_coords[:, 0].cpu().numpy()
    lons1 = pred_coords[:, 1].cpu().numpy()
    lats2 = true_coords[:, 0].cpu().numpy()
    lons2 = true_coords[:, 1].cpu().numpy()

    # batch‑compute forward/inverse geodesic
    # returns: az12, az21, distance_in_meters
    _, _, dist_m = _geod.inv(lons1, lats1, lons2, lats2)

    # back to torch, convert to km
    return torch.from_numpy(dist_m / 1000.0).to(pred_coords.device)

def compute_metrics_geo(pred_coords, true_coords, thresholds=[1,25,200,750,2500]):
    """
    Uses the PROJ‑based vectorized geodesic distance.
    """
    distances = geodesic_distance_vectorized(pred_coords, true_coords)
    metrics = {}
    for thresh in thresholds:
        metrics[f"within_{thresh}km"] = (distances <= thresh).float().mean().item() * 100
    return metrics

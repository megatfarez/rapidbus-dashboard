# app.py — Rapid Bus Bunching (direction-aware + along-path distances + Google Map)
# Minimal single-file Streamlit app. Requires vendor_gtfs_rt/gtfs_realtime_pb2.py in the project.

import os, math, time, random, datetime, uuid
import pandas as pd
import requests
import streamlit as st
import uuid
import json
from streamlit_folium import st_folium
import folium
from streamlit import components
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

from vendor_gtfs_rt import gtfs_realtime_pb2


# ---------- CONFIG ----------
GTFS_URL = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana/?category=rapid-bus-kl"
TIMEOUT_S = 15

st.set_page_config(page_title="Rapid Bus Bunching", layout="wide")


# ---------- TIME HELPERS ----------
def fmt_ts(ts):
    try:
        ts = int(ts)
    except Exception:
        return ""
    if ZoneInfo:
        return datetime.datetime.fromtimestamp(ts, tz=ZoneInfo("Asia/Kuala_Lumpur")).strftime("%Y-%m-%d %H:%M:%S %Z")
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


# ---------- MATH / GEO ----------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.0)**2
    return 2.0 * R * math.asin(math.sqrt(a))

def _to_xy(lat, lon, lat0, lon0):
    lat_r, lon_r, lat0_r, lon0_r = map(math.radians, [lat, lon, lat0, lon0])
    x = (lon_r - lon0_r) * math.cos((lat_r + lat0_r) * 0.5) * 6371000.0
    y = (lat_r - lat0_r) * 6371000.0
    return x, y

def snap_progress_m(shape, lat, lon):
    xs, ys, cum = shape["xs"], shape["ys"], shape["cum_m"]
    lat0, lon0 = shape["ref"]
    px, py = _to_xy(lat, lon, lat0, lon0)
    best_d2, best_prog = float("inf"), 0.0
    for i in range(len(xs) - 1):
        ax, ay = xs[i], ys[i]
        bx, by = xs[i+1], ys[i+1]
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        seg2 = vx*vx + vy*vy
        t = 0.0 if seg2 <= 1e-9 else max(0.0, min(1.0, (vx*wx + vy*wy) / seg2))
        qx, qy = ax + t*vx, ay + t*vy
        dx, dy = px - qx, py - qy
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            best_prog = cum[i] + t * math.hypot(vx, vy)
    return best_prog, math.sqrt(best_d2)


# ---------- GTFS STATIC (optional; for along-path distances) ----------
def load_gtfs_static(zip_path):
    """Reads trips.txt & shapes.txt from a GTFS static zip and builds in-memory structures."""
    if not zip_path or not os.path.exists(zip_path):
        return None
    import zipfile, io, csv
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        if "trips.txt" not in names or "shapes.txt" not in names:
            return None

        def read_csv(name):
            with zf.open(name, "r") as f:
                return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")))

        trips_rows = read_csv("trips.txt")
        shapes_rows = read_csv("shapes.txt")

        trips = {}                 # trip_id -> shape_id
        route_dir_to_shapes = {}   # (route_id, dir_key) -> set(shape_id)
        trip_dir = {}              # trip_id -> direction_id (0/1)

        for r in trips_rows:
            trip_id = r.get("trip_id", "")
            route_id = r.get("route_id", "")
            shape_id = r.get("shape_id", "")
            d = r.get("direction_id", "")
            dir_key = "NA" if not d else str(int(float(d)))
            if trip_id and shape_id:
                trips[trip_id] = shape_id
            if route_id and shape_id:
                route_dir_to_shapes.setdefault((route_id, dir_key), set()).add(shape_id)
            if trip_id and d != "":
                try:
                    trip_dir[trip_id] = int(float(d))
                except Exception:
                    pass

        by_sid = {}
        for r in shapes_rows:
            sid = r.get("shape_id", "")
            if not sid:
                continue
            la = float(r.get("shape_pt_lat", "0") or 0)
            lo = float(r.get("shape_pt_lon", "0") or 0)
            seq = int(float(r.get("shape_pt_sequence", "0") or 0))
            by_sid.setdefault(sid, []).append((seq, la, lo))

        shapes = {}
        for sid, pts in by_sid.items():
            pts.sort(key=lambda x: x[0])
            lats = [p[1] for p in pts]
            lons = [p[2] for p in pts]
            if len(lats) < 2:
                continue
            lat0, lon0 = lats[0], lons[0]
            xs, ys = zip(*[_to_xy(la, lo, lat0, lon0) for la, lo in zip(lats, lons)])
            cum = [0.0]
            for i in range(len(xs) - 1):
                seg = math.hypot(xs[i+1] - xs[i], ys[i+1] - ys[i])
                cum.append(cum[-1] + seg)
            shapes[sid] = {"xs": list(xs), "ys": list(ys), "cum_m": cum, "ref": (lat0, lon0)}

        return {"trips": trips, "route_dir_to_shapes": route_dir_to_shapes, "shapes": shapes, "trip_dir": trip_dir}

@st.cache_resource(show_spinner=False)
def get_static_gtfs(zip_path=None):
    if zip_path is None:
        zip_path = os.getenv("GTFS_STATIC_ZIP", "data/gtfs_static.zip")
    try:
        return load_gtfs_static(zip_path)
    except Exception:
        return None

def pick_shape_for_group(static, route_id, dir_key, samples):
    """Pick the best shape (lowest mean snap residual) for route+direction group."""
    if static is None:
        return None, None
    cand = static["route_dir_to_shapes"].get((str(route_id), str(dir_key))) or \
           static["route_dir_to_shapes"].get((str(route_id), "NA"))
    if not cand:
        return None, None
    best_sid, best_mean = None, None
    for sid in cand:
        shape = static["shapes"].get(sid)
        if not shape:
            continue
        total = 0.0
        n = 0
        for la, lo in samples:
            _, res = snap_progress_m(shape, la, lo)
            total += res
            n += 1
        if n == 0:
            continue
        mean_res = total / n
        if best_mean is None or mean_res < best_mean:
            best_mean, best_sid = mean_res, sid
    return best_sid, best_mean


# ---------- DATA FETCH ----------
def fetch_gtfs(max_retries=5):
    headers = {"User-Agent": "rapidbus-bunching/1.0"}
    last_error = None
    for i in range(max_retries):
        try:
            r = requests.get(GTFS_URL, headers=headers, timeout=TIMEOUT_S)
            if r.status_code == 429:
                wait = r.headers.get("Retry-After")
                sleep_s = int(wait) if (wait and str(wait).isdigit()) else (2 ** i) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            last_error = e
            time.sleep((2 ** i) + random.uniform(0, 0.5))
    raise last_error if last_error else RuntimeError("Unknown error fetching GTFS")

@st.cache_data(ttl=45, show_spinner=False)
def get_gtfs_bytes():
    return fetch_gtfs()


# ---------- GOOGLE MAPS KEY HELPER ----------
def _get_gmaps_key():
    # priority: session_state -> secrets -> env
    k = st.session_state.get("gmaps_key")
    if not k:
        try:
            k = st.secrets.get("GOOGLE_MAPS_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            k = None
    if not k:
        k = os.getenv("GOOGLE_MAPS_API_KEY")
    return k


# ---------- LOAD REALTIME VEHICLE POSITIONS ----------
def load_vehicle_positions_df():
    feed_bytes = get_gtfs_bytes()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(feed_bytes)
    rows = []
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        vp = entity.vehicle
        trip = vp.trip if vp.HasField("trip") else None

        # Safe presence check for direction_id (avoid treating default 0 as real)
        direction_id = None
        if trip is not None:
            try:
                if trip.HasField("direction_id"):
                    direction_id = int(trip.direction_id)
            except Exception:
                direction_id = None

        try:
            rows.append({
                "vehicle_id": str(getattr(vp.vehicle, "id", "")),
                "route_id":   str(getattr(vp.trip, "route_id", "")),
                "trip_id":    str(getattr(vp.trip, "trip_id", "")),
                "lat":        float(getattr(vp.position, "latitude", 0.0)),
                "lon":        float(getattr(vp.position, "longitude", 0.0)),
                "timestamp":  int(vp.timestamp) if getattr(vp, "timestamp", 0) else None,
                "bus_plate":  getattr(vp.vehicle, "label", "") or getattr(vp.vehicle, "license_plate", ""),
                "direction_id": direction_id,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


# ---------- ENRICH / DETECT BUNCHING ----------
def enrich_bunching(df: pd.DataFrame, max_sep_m: float = 200.0) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()

    # Fill direction from GTFS static if available (trips.txt is authoritative)
    static = get_static_gtfs()
    if static and "trip_dir" in static and "trip_id" in df.columns:
        td = static["trip_dir"]
        df["direction_id"] = df.apply(
            lambda r: td.get(str(r["trip_id"]), r["direction_id"]),
            axis=1
        )

    # Group by route and direction (if present)
    if "direction_id" in df.columns:
        df["_dir_key"] = df["direction_id"].apply(lambda v: "NA" if pd.isna(v) else str(int(v)))
        groupby_keys = [df["route_id"].astype(str), df["_dir_key"]]
    else:
        groupby_keys = [df["route_id"].astype(str)]

    # Working columns
    df["nn_vehicle_id"] = pd.NA
    df["nn_distance_m"] = pd.NA
    df["sep_mode"] = pd.NA
    df["sep_mode_label"] = pd.NA
    df["prog_m"] = pd.NA
    df["prog_m_trip"] = pd.NA
    df["shape_id"] = pd.NA
    df["path_quality_m"] = pd.NA

    for _, g in df.groupby(groupby_keys):
        idxs = list(g.index)

        # Choose a canonical shape per group for along-path progress
        group_shape = None
        if static is not None and len(idxs) >= 2:
            arow = df.loc[idxs[0]]
            a_dir = arow.get("direction_id") if "direction_id" in df.columns else None
            dir_key = "NA" if (a_dir is None or (isinstance(a_dir, float) and pd.isna(a_dir))) else str(int(a_dir))
            samples = [(float(df.loc[k, "lat"]), float(df.loc[k, "lon"])) for k in idxs]
            sid, mean_res = pick_shape_for_group(static, str(arow.get("route_id")), dir_key, samples)
            if sid and sid in static["shapes"]:
                group_shape = static["shapes"][sid]
                for i in idxs:
                    df.at[i, "path_quality_m"] = float(mean_res)

        # Compute group progress along canonical shape
        if group_shape is not None:
            for i in idxs:
                prog, _ = snap_progress_m(group_shape, float(df.at[i, "lat"]), float(df.at[i, "lon"]))
                df.at[i, "prog_m"] = float(prog)

        # Trip-aware progress (when we can resolve shape by trip)
        if static is not None and "trip_id" in df.columns:
            for i in idxs:
                t_id = str(df.at[i, "trip_id"]) if pd.notna(df.at[i, "trip_id"]) else ""
                sid2 = static["trips"].get(t_id)
                if sid2 and sid2 in static["shapes"]:
                    prog2, _ = snap_progress_m(static["shapes"][sid2], float(df.at[i, "lat"]), float(df.at[i, "lon"]))
                    df.at[i, "prog_m_trip"] = float(prog2)
                    df.at[i, "shape_id"] = sid2

        # Nearest-neighbour within the group
        for i in idxs:
            a = df.loc[i]
            best_dist, best_vid, best_mode = None, None, None
            for j in idxs:
                if j == i:
                    continue
                b = df.loc[j]
                if pd.notna(a.get("prog_m")) and pd.notna(b.get("prog_m")):
                    d = abs(float(a.get("prog_m")) - float(b.get("prog_m")))
                    mode = "PATH"
                elif pd.notna(a.get("prog_m_trip")) and pd.notna(b.get("prog_m_trip")) and a.get("shape_id") == b.get("shape_id"):
                    d = abs(float(a.get("prog_m_trip")) - float(b.get("prog_m_trip")))
                    mode = "PATH"
                else:
                    d = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
                    mode = "STRAIGHT"
                if best_dist is None or d < best_dist:
                    best_dist, best_vid, best_mode = d, b["vehicle_id"], mode
            if best_vid is not None:
                df.at[i, "nn_vehicle_id"] = str(best_vid)
                df.at[i, "nn_distance_m"] = float(best_dist)
                df.at[i, "sep_mode"] = best_mode
                df.at[i, "sep_mode_label"] = "PATH ✅" if best_mode == "PATH" else "STRAIGHT ➖"

    # Risk/score
    def risk_from_sep(x):
        if pd.isna(x):
            return "LOW"
        return "HIGH" if x <= max_sep_m else "LOW"

    def score_from_sep(x):
        if pd.isna(x):
            return 0
        cap = max_sep_m * 2.0
        return round(max(0.0, (cap - min(cap, float(x))) / cap) * 100.0)

    df["risk"] = df["nn_distance_m"].apply(risk_from_sep)
    df["score"] = df["nn_distance_m"].apply(score_from_sep)
    return df


def get_pair_for_vehicle(df: pd.DataFrame, vid: str):
    if df is None or df.empty or not vid:
        return None, None
    cand = df.loc[df["vehicle_id"].astype(str) == str(vid)]
    if cand.empty:
        return None, None
    a = cand.iloc[0]
    if "nn_vehicle_id" in df.columns and pd.notna(a.get("nn_vehicle_id")):
        nb = df.loc[df["vehicle_id"].astype(str) == str(a.get("nn_vehicle_id"))]
        if not nb.empty:
            return a, nb.iloc[0]
    same = (df["route_id"].astype(str) == str(a.get("route_id"))) & (df["vehicle_id"].astype(str) != str(vid))
    if "direction_id" in df.columns:
        a_dir = a.get("direction_id")
        if pd.notna(a_dir):
            same &= (df["direction_id"] == a_dir)
    same_group = df.loc[same].copy()
    if same_group.empty:
        return a, None

    def _sep(row):
        if "prog_m" in df.columns and pd.notna(a.get("prog_m")) and pd.notna(row.get("prog_m")):
            return abs(float(a.get("prog_m")) - float(row.get("prog_m")))
        return haversine_m(a.get("lat"), a.get("lon"), row.get("lat"), row.get("lon"))

    same_group["__dist_m"] = same_group.apply(_sep, axis=1)
    b = same_group.nsmallest(1, "__dist_m").iloc[0]
    return a, b


# ---------- VISUALS ----------
def render_bunching_strip(a, b, threshold_m=200.0):
    if a is None:
        st.info("Select a bus from the table to view the strip.")
        return
    sep_m = a.get("nn_distance_m")
    if (sep_m is None or (isinstance(sep_m, float) and math.isnan(sep_m))) and b is not None:
        sep_m = haversine_m(float(a.get("lat")), float(a.get("lon")), float(b.get("lat")), float(b.get("lon")))
    if sep_m is None or (isinstance(sep_m, float) and math.isnan(sep_m)):
        st.warning("No neighbour found on the same route to visualize.")
        return
    cap = float(threshold_m) * 2.0
    clamped = min(float(sep_m), cap)
    xB = clamped / cap
    vehicle_a = str(a.get("vehicle_id"))
    vehicle_b = str(b.get("vehicle_id")) if b is not None else "—"
    route = str(a.get("route_id"))
    plate_a = a.get("bus_plate", "")
    plate_b = b.get("bus_plate", "") if b is not None else ""
    risk = "HIGH" if float(sep_m) <= float(threshold_m) else "LOW"

    html = f"""
    <style>
    .strip {{position:relative;height:16px;border-radius:8px;background:#eee;overflow:hidden;}}
    .tick {{position:absolute;top:100%;transform:translateX(-50%);font-size:12px;color:#666;margin-top:6px;}}
    .dot {{position:absolute;top:50%;transform:translate(-50%,-50%);width:18px;height:18px;border-radius:50%;}}
    .dotA {{background:#444;}} .dotB {{background:#0a7;}}
    .label-row {{display:flex;justify-content:space-between;gap:12px;margin-bottom:6px;font-size:14px;}}
    .badge-high {{border:1px solid #f2c0c0;color:#a00;background:#fff5f5;padding:2px 8px;border-radius:999px}}
    .badge-low  {{border:1px solid #cfe8d9;color:#065;background:#f4fbf7;padding:2px 8px;border-radius:999px}}
    </style>
    <div class="label-row">
      <div><b>Route:</b> {route} &nbsp;&nbsp; <span class="{ 'badge-high' if risk=='HIGH' else 'badge-low' }">{risk} risk</span></div>
      <div><b>Separation:</b> {sep_m:.0f} m &nbsp;&nbsp; (thr: {threshold_m:.0f} m)</div>
    </div>
    <div class="strip">
      <div class="tick" style="left:{threshold_m/cap*100:.4f}%;">|<br/><span style="font-size:11px">thr</span></div>
      <div class="dot dotA" style="left:0%;"></div>
      <div class="dot dotB" style="left:{xB*100:.4f}%;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:13px;color:#333;">
      <div><b>A</b> • {vehicle_a} {('('+plate_a+')' if plate_a else '')}</div>
      <div><b>B</b> • {vehicle_b} {('('+plate_b+')' if plate_b else '')}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_gmap_pair(a, b, api_key: str, height_px: int = 320):
    """Embed a Google Map for the pair (A,B). Doubled braces are used for JS object literals.
    Includes a JS error-listener that writes errors into the map DIV for easy debugging.
    """
    if not api_key:
        st.info("Provide a Google Maps API key (set env `GOOGLE_MAPS_API_KEY`, Streamlit secrets, or enter it below).")
        return

    # Coordinates
    try:
        la1, lo1 = float(a.get("lat")), float(a.get("lon"))
    except Exception:
        st.warning("Missing coordinates for vehicle A")
        return

    has_b = b is not None
    if has_b:
        try:
            la2, lo2 = float(b.get("lat")), float(b.get("lon"))
        except Exception:
            has_b = False

    # Labels/info
    vid_a = str(a.get("vehicle_id", "A"))
    vid_b = str(b.get("vehicle_id")) if has_b else ""
    route = str(a.get("route_id", ""))
    sep = a.get("nn_distance_m")
    sep_txt = f"{float(sep):.0f} m" if sep is not None else "–"

    # Map ids
    mid_lat = la1 if not has_b else (la1 + la2) / 2.0
    mid_lng = lo1 if not has_b else (lo1 + lo2) / 2.0
    map_id = "map_" + "%08x" % (uuid.uuid4().int & 0xFFFFFFFF)
    init_fn = "initMap_" + "%08x" % (uuid.uuid4().int & 0xFFFFFFFF)

    # Optional JS for bus B
    marker_b = ""
    bounds_b = ""
    line_js = ""
    info_b = ""
    if has_b:
        safe_vid_b = vid_b.replace("'", "\\'")
        safe_route = route.replace("'", "\\'")
        marker_b = (
            f"const b = new google.maps.Marker({{position: {{lat: {la2}, lng: {lo2}}}, map: map, "
            f"label: 'B', title: 'B: {safe_vid_b} (Route {safe_route})'}});"
        )
        bounds_b = "bounds.extend(b.getPosition());"
        line_js = (
            f"const line = new google.maps.Polyline({{path: [{{lat: {la1}, lng: {lo1}}}, "
            f"{{lat: {la2}, lng: {lo2}}}], geodesic: true, strokeOpacity: 1.0, strokeWeight: 3}}); "
            "line.setMap(map);"
        )
        info_b = (
            f"const iwB = new google.maps.InfoWindow({{content: 'B: {safe_vid_b}<br/>Route {safe_route}<br/>Separation {sep_txt}'}}); "
            "b.addListener('click', () => iwB.open({anchor: b, map}));"
        )

    html = f"""
    <div id="{map_id}" style="width:100%;height:{height_px}px;border-radius:12px;overflow:hidden;border:1px solid #eee;"></div>
    <script>
    function {init_fn}() {{
        const map = new google.maps.Map(document.getElementById("{map_id}"), {{
            zoom: 13,
            center: {{lat: {mid_lat}, lng: {mid_lng}}},
            mapTypeId: 'roadmap',
        }});
        const a = new google.maps.Marker({{
            position: {{lat: {la1}, lng: {lo1}}},
            map: map,
            label: 'A',
            title: 'A: {vid_a} (Route {route})'
        }});
        let bounds = new google.maps.LatLngBounds();
        bounds.extend(a.getPosition());
        {marker_b}
        {bounds_b}
        {line_js}
        map.fitBounds(bounds);
        const iwA = new google.maps.InfoWindow({{content: "A: {vid_a}<br/>Route {route}"}}); 
        a.addListener('click', () => iwA.open({{anchor: a, map}}));
        {info_b}
    }}
    </script>
    <script>
    // Surface JS errors visibly on the page (helps diagnose key/referer issues)
    window.addEventListener('error', function(e) {{
      try {{
        const el = document.getElementById("{map_id}");
        if (el && e && e.message) {{
          el.innerHTML = '<div style="padding:12px;color:#a00;font-family:monospace">'
            + 'JS error: ' + e.message + '</div>';
        }}
      }} catch (_) {{ }}
    }});
    </script>
    <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback={init_fn}" async defer></script>
    """
    components.v1.html(html, height=height_px + 10)


# ---------- SIDEBAR ----------
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Bunching separation threshold (m)", 50, 500, 200, 10)
only_path = st.sidebar.checkbox("Show only PATH-mode flags", value=False)
auto_refresh = st.sidebar.checkbox("Auto refresh", value=False)
if auto_refresh:
    st.sidebar.caption("Refresh interval ≈ 60s to respect API rate limits.")
    st.sidebar.write(f"Last refresh: {time.strftime('%H:%M:%S')}")

# Optional: allow entering key via sidebar
with st.sidebar.expander("Google Map"):
    _k0 = _get_gmaps_key()
    _k1 = st.text_input(
        "Google Maps API key",
        value=_k0 or "",
        type="password",
        help="You can also set env GOOGLE_MAPS_API_KEY or add it to Streamlit secrets.",
        key="gmaps_key_sidebar",
    )
    if _k1:
        st.session_state["gmaps_key"] = _k1


# ---------- MAIN FLOW ----------
try:
    df_raw = load_vehicle_positions_df()
    st.session_state["last_df"] = df_raw
except Exception as e:
    st.warning(f"Rate limited / network issue. Showing last known data if available. Details: {e}")
    df_raw = st.session_state.get("last_df")
    if df_raw is None:
        st.stop()

enriched = enrich_bunching(df_raw, max_sep_m=float(threshold))
if "timestamp" in enriched.columns:
    enriched["local_time"] = enriched["timestamp"].apply(fmt_ts)

flags = enriched.loc[enriched["risk"] == "HIGH"] if enriched is not None else pd.DataFrame()
if only_path and not flags.empty:
    flags = flags.loc[flags["sep_mode"] == "PATH"]
if not flags.empty:
    flags = flags.sort_values(by=["score", "nn_distance_m"], ascending=[False, True]).reset_index(drop=True)

st.title("Rapid Bus Bunching")

# last updated (safe + robust)
if not enriched.empty and "timestamp" in enriched.columns:
    try:
        ts_series = pd.to_numeric(enriched["timestamp"], errors="coerce").dropna()
        if not ts_series.empty:
            ts = int(ts_series.max())
            st.caption("Last vehicle timestamp: " + fmt_ts(ts))
    except Exception:
        pass


# ---------- FLAGS TABLE + ROW ACTIONS ----------
st.subheader("Flagged buses")
if flags is not None and not flags.empty:
    show_cols = [c for c in [
        "local_time", "route_id", "direction_id", "vehicle_id", "bus_plate",
        "risk", "score", "nn_vehicle_id", "nn_distance_m", "sep_mode_label", "path_quality_m"
    ] if c in flags.columns]
    st.dataframe(flags[show_cols].rename(columns={"nn_distance_m": "sep_m"}),
                 use_container_width=True, hide_index=True)

    st.markdown("**Click Inspect to open the straight-line view, or show a quick map:**")
    for i, r in flags.iterrows():
        c1, c2, c3, c4, c5, c6 = st.columns([3, 3, 3, 3, 2, 2])
        c1.write(f"Route **{r.get('route_id','–')}**")
        c2.write(f"Vehicle **{r.get('vehicle_id','–')}**")
        c3.write(f"Risk **{r.get('risk','–')}** | Score **{int(r.get('score',0))}**")
        sep = r.get("nn_distance_m")
        c4.write(f"Separation **{sep:.0f} m**" if pd.notna(sep) else "Separation –")

        # Unique, stable key per row
        rowkey = f"{i}_{str(r.get('vehicle_id'))}_{str(r.get('route_id'))}_{str(r.get('nn_vehicle_id'))}_{int(r.get('timestamp') or 0)}"

        if c5.button("🔍 Inspect", key=f"inspect_{rowkey}"):
            st.session_state["selected_vehicle"] = str(r.get("vehicle_id"))

        if c6.button("🗺 Map", key=f"map_{rowkey}"):
            gk = _get_gmaps_key()
            if not gk:
                st.warning("Google Maps API key is missing.")
                gk_inline = st.text_input(
                    "Google Maps API key",
                    value="",
                    type="password",
                    key=f"gmaps_inline_{rowkey}",
                    help="Paste your key here (or set env GOOGLE_MAPS_API_KEY / Streamlit secrets)."
                )
                if gk_inline:
                    st.session_state["gmaps_key"] = gk_inline
                    try:
                        st.rerun()
                    except Exception:
                        pass
            else:
                nb = enriched.loc[enriched["vehicle_id"].astype(str) == str(r.get("nn_vehicle_id"))]
                b = nb.iloc[0] if not nb.empty else None
                render_gmap_pair(r, b, gk, height_px=280)
else:
    st.info("No current flags.")


# ---------- FOCUS PANEL ----------
st.markdown("---")
st.subheader("Bunching Focus")
sel_vid = st.session_state.get("selected_vehicle")
if not sel_vid:
    st.caption("Use **🔍 Inspect** in the table to open the straight-line view.")
else:
    a, b = get_pair_for_vehicle(enriched, sel_vid)
    if a is None:
        st.warning("Could not find the selected bus in the current frame.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vehicle A", str(a.get("vehicle_id")))
        c2.metric("Route", str(a.get("route_id")))
        c3.metric("Risk", str(a.get("risk")))
        sep_val = a.get("nn_distance_m")
        c4.metric("Separation", f"{float(sep_val):.0f} m" if sep_val is not None else "–")

        using_path = (str(a.get("sep_mode")) == "PATH")
        # Timestamp headway (Δtime) between A and B
        dt_label = "–"
        ta = a.get("timestamp")
        tb = b.get("timestamp") if b is not None else None
        if ta and tb:
            try:
                dt = abs(int(ta) - int(tb))
                m_, s_ = divmod(dt, 60)
                dt_label = f"{m_}m {s_}s" if m_ else f"{s_}s"
            except Exception:
                pass
        st.caption(f"Separation mode: {'PATH (along route)' if using_path else 'STRAIGHT-LINE'}" + f" | Δtime: {dt_label}")
        render_bunching_strip(a, b, threshold_m=float(threshold))

        # Inline Google Map prompt/render
        gk = _get_gmaps_key()
        if not gk:
            st.warning("Google Maps API key is missing.")
            gk_inline_focus = st.text_input(
                "Google Maps API key",
                value="",
                type="password",
                key="gmaps_key_focus",
                help="Paste your key here (or set env GOOGLE_MAPS_API_KEY / Streamlit secrets)."
            )
            if gk_inline_focus:
                st.session_state["gmaps_key"] = gk_inline_focus
                try:
                    st.rerun()
                except Exception:
                    pass
        else:
            st.caption('Google Map')
            render_gmap_pair(a, b, gk)

        if st.button("Clear selection"):
            st.session_state["selected_vehicle"] = None



def render_gmap_overview(flags_df, enriched_df, api_key: str, height_px: int = 520):
    """One big Google Map showing all flagged pairs (A markers & line to nearest B) via /static page (no srcdoc)."""
    import streamlit as st

    if not api_key:
        st.info("Provide a Google Maps API key (env `GOOGLE_MAPS_API_KEY`, Streamlit secrets, or enter below).")
        return
    if flags_df is None or flags_df.empty:
        st.info("No flagged buses to display on the map.")
        return

    # Build points and segments
    pts, lines, seen = [], [], set()
    for _, r in flags_df.iterrows():
        try:
            la1, lo1 = float(r.get("lat")), float(r.get("lon"))
        except Exception:
            continue
        vid_a = str(r.get("vehicle_id"))
        route = str(r.get("route_id", ""))
        if vid_a not in seen:
            pts.append({"id": vid_a, "lat": la1, "lon": lo1, "label": "A", "title": f"A: {vid_a} (Route {route})"})
            seen.add(vid_a)

        nn = str(r.get("nn_vehicle_id"))
        if nn and nn != "nan":
            nb = enriched_df.loc[enriched_df["vehicle_id"].astype(str) == nn]
            if not nb.empty:
                rb = nb.iloc[0]
                try:
                    la2, lo2 = float(rb.get("lat")), float(rb.get("lon"))
                except Exception:
                    la2 = lo2 = None
                if la2 is not None and lo2 is not None:
                    if nn not in seen:
                        pts.append({"id": nn, "lat": la2, "lon": lo2, "label": "B", "title": f"B: {nn} (Route {route})"})
                        seen.add(nn)
                    lines.append((la1, lo1, la2, lo2))

    if not pts:
        st.info("No valid coordinates to plot.")
        return

    mid_lat = sum(p["lat"] for p in pts) / len(pts)
    mid_lng = sum(p["lon"] for p in pts) / len(pts)
    init_fn = "initMap_" + "%08x" % (uuid.uuid4().int & 0xFFFFFFFF)

    pts_js   = json.dumps(pts)
    lines_js = json.dumps(lines)

    # Build a small standalone HTML page (served from /static), so referrer is http://localhost:8501/static/...
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Map Overview</title>
<style>
  html, body, #host {{ height: 100%; margin: 0; padding: 0; }}
  #host {{ position: relative; border-radius: 12px; overflow: hidden; background:#f6f6f6; }}
  .badge {{
    position:absolute; right:6px; bottom:6px; padding:2px 6px; border-radius:6px;
    background:rgba(0,0,0,.55); color:#fff; font:12px/16px monospace;
  }}
</style>
</head>
<body>
  <div id="host"><div class="badge" id="badge">maps: init…</div></div>
  <script>
  function {init_fn}() {{
    const host = document.getElementById("host");
    const badge = document.getElementById("badge");
    const map = new google.maps.Map(host, {{
      zoom: 12,
      center: {{lat: {mid_lat}, lng: {mid_lng}}},
      mapTypeId: 'roadmap',
    }});
    const pts = {pts_js};
    const lines = {lines_js};
    let bounds = new google.maps.LatLngBounds();
    for (const p of pts) {{
      const m = new google.maps.Marker({{
        position: {{lat: p.lat, lng: p.lon}},
        map: map,
        label: p.label || '',
        title: p.title || p.id
      }});
      bounds.extend(m.getPosition());
    }}
    for (const seg of lines) {{
      const line = new google.maps.Polyline({{
        path: [{{lat: seg[0], lng: seg[1]}}, {{lat: seg[2], lng: seg[3]}}],
        geodesic: true, strokeOpacity: 1.0, strokeWeight: 3
      }});
      line.setMap(map);
    }}
    map.fitBounds(bounds);
    google.maps.event.addListenerOnce(map, 'idle', () => {{ badge.textContent = 'maps: idle'; }});
  }}
  // Inline JS error -> visible message
  window.addEventListener('error', function(e) {{
    try {{
      const el = document.getElementById("host");
      if (el && e && e.message) {{
        el.innerHTML = '<div style="padding:12px;color:#a00;font-family:monospace">JS error: ' + e.message + '</div>';
      }}
    }} catch (_) {{ }}
  }});
  </script>
  <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback={init_fn}" async defer></script>
</body>
</html>"""

    # Write to ./static/gmap_overview.html
    base_dir   = os.path.dirname(__file__)
    static_dir = os.path.join(base_dir, "static")
    os.makedirs(static_dir, exist_ok=True)
    out_path = os.path.join(static_dir, "gmap_overview.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Embed the file by URL (no srcdoc)
    components.v1.iframe(src="static/gmap_overview.html", height=height_px, scrolling=False)


import numpy as np
import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout

# BEST PRACTICE: Prevent local pop-up windows from interrupting the web server
pv.OFF_SCREEN = True

# Initialize Trame Server
server = get_server()
state, ctrl = server.state, server.controller

# Setup global game data
attempts = []

# --- LOAD DATA ---
mesh = pv.Sphere(radius=10.0)  
gt_point = np.array([0.0, 0.0, 10.0])
distance = 2.45  
lmk = "Nasion"

# Setup Plotter for Web
pl = pv.Plotter(window_size=[800, 600])
pl.add_mesh(mesh, color="#E9A76E", show_edges=False)
pl.add_title(f"Landmarking Game: Tap the {lmk} landmark!", font_size=12)

# Define tap callback logic for iPad
def game_callback(picked_point):
    if picked_point is None:
        return
    
    distance_u = np.linalg.norm(picked_point - gt_point)
    attempts.append(picked_point)
        
    if distance_u < distance:
        color = "green"
        msg = f"🎯 Better than the model! Error: {distance_u:.2f} mm.\n{len(attempts)} attempts!"
    elif distance_u < 5:
        color = "yellow"
        msg = f"🎯 Good! Error: {distance_u:.2f} mm.\nBeat the model error: {distance:.2f} mm! \n{len(attempts)} attempts!"
    else:
        color = "red"
        msg = f"❌ Missed! Error: {distance_u:.2f} mm. Try again! \n{len(attempts)} attempts!"
        
    marker = pv.Sphere(radius=0.3, center=picked_point)
    pl.add_mesh(marker, color=color, name=f"attempt_{len(attempts)}")
    pl.add_title(msg, font_size=12)
    
    # Force the web view to update/re-render immediately on the iPad screen
    ctrl.view_update()

# Build Trame Web Layout
with SinglePageLayout(server) as layout:
    layout.title.text = "3D Landmarking Challenge"
    with layout.content:
        # FIX: Set mode to "server" to bypass the early local geometry update crash
        view = plotter_ui(pl, mode="server")
        ctrl.view_update = view.update

# Wait until the server protocol officially exists to set up the picker callback
@ctrl.add("on_server_ready")
def setup_picking(**kwargs):
    pl.enable_point_picking(
        callback=game_callback, 
        show_message=False, 
        color="blue",
        picker='cell', 
        pickable_window=True
    )

# Start application server
if __name__ == "__main__":
    server.start(host="0.0.0.0", port=8080)
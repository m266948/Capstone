import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import threading
import time
import datetime
import queue
from collections import OrderedDict
import tkintermapview
import random
import zipfile
import os
from tkinter import filedialog, simpledialog, messagebox
import tkinter as tk
import tkinter.ttk as ttk

# --- Helper Class: Centroid Tracker ---
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict() 
        self.disappeared = OrderedDict() 
        self.max_disappeared = max_disappeared

    def register(self, centroid, box):
        self.objects[self.next_object_id] = (centroid, box)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def reset(self):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, w, h)) in enumerate(rects):
            cX = int(startX + w / 2.0)
            cY = int(startY + h / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(rects)):
                self.register(input_centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_values = list(self.objects.values())
            object_centroids = np.array([v[0] for v in object_values])

            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row][col] > 100: continue

                object_id = object_ids[row]
                self.objects[object_id] = (input_centroids[col], rects[col]) 
                self.disappeared[object_id] = 0 

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], rects[col])

        return self.objects

# --- Helper Class: Map Marker Manager ---
class MapMarkerManager:
    """Manages markers on the topographic map with filtering and visibility control."""
    
    def __init__(self, map_widget, center_lat, center_lon):
        self.map_widget = map_widget
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.markers = OrderedDict()  # {marker_id: {'marker_obj': obj, 'affiliation': str, 'data': dict, 'visible': bool}}
        self.next_marker_id = 0
        self.deletion_history = []  # For undo functionality
        
        # Affiliation color mapping
        self.affiliation_colors = {
            'Hostile': '#ff0000',      # Red
            'Friendly': '#00ff00',     # Green
            'Neutral': '#ffff00',      # Yellow
            'Unknown': '#0080ff',      # Blue
            'Civilian': '#ffffff'      # White
        }
    
    def get_random_coords(self):
        """Generate random coordinates within ±0.01 degrees of map center for demo."""
        lat_offset = random.uniform(-0.01, 0.01)
        lon_offset = random.uniform(-0.01, 0.01)
        return (self.center_lat + lat_offset, self.center_lon + lon_offset)
    
    def add_marker(self, affiliation='Unknown', detection_data=None, coords=None):
        """Add a new marker to the map."""
        if coords is None:
            coords = self.get_random_coords()
        
        color = self.affiliation_colors.get(affiliation, '#0080ff')
        
        # Create marker text
        marker_text = f"{affiliation}\nID:{self.next_marker_id}"
        
        try:
            marker_obj = self.map_widget.set_marker(
                coords[0], coords[1],
                text=marker_text,
                marker_color_circle=color,
                marker_color_outside='gray'
            )
            
            self.markers[self.next_marker_id] = {
                'marker_obj': marker_obj,
                'affiliation': affiliation,
                'data': detection_data or {},
                'visible': True,
                'coords': coords
            }
            
            self.next_marker_id += 1
            return self.next_marker_id - 1
            
        except Exception as e:
            print(f"Error adding marker: {e}")
            return None
    
    def remove_marker(self, marker_id):
        """Remove a marker by ID and add to deletion history."""
        if marker_id in self.markers:
            marker_data = self.markers[marker_id]
            try:
                marker_data['marker_obj'].delete()
            except:
                pass
            self.deletion_history.append(marker_data)
            if len(self.deletion_history) > 10:
                self.deletion_history.pop(0)
            del self.markers[marker_id]
    
    def clear_all_markers(self):
        """Remove all markers from the map."""
        for marker_id in list(self.markers.keys()):
            try:
                self.markers[marker_id]['marker_obj'].delete()
            except:
                pass
        self.markers.clear()
    
    def filter_by_affiliation(self, affiliation, show=True):
        """Show or hide markers based on affiliation."""
        for marker_id, marker_data in self.markers.items():
            if marker_data['affiliation'] == affiliation:
                if show and not marker_data['visible']:
                    # Re-create marker
                    coords = marker_data['coords']
                    color = self.affiliation_colors.get(affiliation, '#0080ff')
                    marker_text = f"{affiliation}\nID:{marker_id}"
                    try:
                        marker_obj = self.map_widget.set_marker(
                            coords[0], coords[1],
                            text=marker_text,
                            marker_color_circle=color,
                            marker_color_outside='gray'
                        )
                        marker_data['marker_obj'] = marker_obj
                        marker_data['visible'] = True
                    except:
                        pass
                elif not show and marker_data['visible']:
                    try:
                        marker_data['marker_obj'].delete()
                        marker_data['visible'] = False
                    except:
                        pass
    
    def get_marker_count_by_affiliation(self):
        """Return count of markers by affiliation."""
        counts = {}
        for marker_data in self.markers.values():
            aff = marker_data['affiliation']
            counts[aff] = counts.get(aff, 0) + 1
        return counts
    
    def get_all_coords(self):
        """Get all marker coordinates for bounding box calculation."""
        return [marker_data['coords'] for marker_data in self.markers.values() if marker_data['visible']]

# --- Helper Class: Drawing Tools Manager ---
class DrawingToolsManager:
    """Manages custom drawn objects (markers, lines, polygons) on the map."""
    
    def __init__(self, map_widget):
        self.map_widget = map_widget
        self.drawn_objects = OrderedDict()  # {obj_id: {'type': str, 'coords': list, 'color': str, 'label': str, 'objects': list}}
        self.next_object_id = 0
        self.drawing_mode = "none"  # none, marker, line, polygon
        self.current_color = "#ff0000"  # Red
        self.temp_points = []  # For line and polygon drawing
        self.temp_markers = []  # Temporary markers during drawing
        self.colors = {
            'Red': '#ff0000',
            'Green': '#00ff00',
            'Blue': '#0080ff',
            'Yellow': '#ffff00',
            'Magenta': '#ff00ff',
            'Cyan': '#00ffff',
            'Orange': '#ff8800',
            'Purple': '#8800ff',
            'White': '#ffffff'
        }
    
    def set_drawing_mode(self, mode):
        """Set the current drawing mode (none, marker, line, polygon)."""
        self.drawing_mode = mode.lower()
        self.temp_points = []
        self.clear_temp_markers()
    
    def set_color(self, color_name):
        """Set the current drawing color."""
        if color_name in self.colors:
            self.current_color = self.colors[color_name]
    
    def clear_temp_markers(self):
        """Remove temporary markers shown during drawing."""
        for marker in self.temp_markers:
            try:
                marker.delete()
            except:
                pass
        self.temp_markers = []
    
    def on_map_click(self, coords):
        """Handle map click for drawing."""
        if self.drawing_mode == "none" or not self.map_widget:
            return False
        
        try:
            lat, lon = coords
            
            if self.drawing_mode == "marker":
                self.add_marker(lat, lon)
                return True
            
            elif self.drawing_mode in ["line", "polygon"]:
                self.temp_points.append((lat, lon))
                
                # Add temporary marker at click point
                if self.map_widget:
                    try:
                        temp_marker = self.map_widget.set_marker(
                            lat, lon,
                            text=f"P{len(self.temp_points)}",
                            marker_color_circle=self.current_color,
                            marker_color_outside='gray'
                        )
                        self.temp_markers.append(temp_marker)
                    except Exception as e:
                        print(f"Error adding temp marker: {e}")
                
                return True
        except Exception as e:
            print(f"Error in on_map_click: {e}")
            return False
        
        return False
    
    def finish_line_or_polygon(self):
        """Finish drawing a line or polygon."""
        if len(self.temp_points) < 2:
            self.temp_points = []
            self.clear_temp_markers()
            return False
        
        if not self.map_widget:
            self.temp_points = []
            self.clear_temp_markers()
            return False
        
        try:
            obj_type = self.drawing_mode
            self.clear_temp_markers()
            self.add_line_or_polygon(obj_type, self.temp_points)
            self.temp_points = []
            return True
        except Exception as e:
            print(f"Error finishing drawing: {e}")
            self.temp_points = []
            return False
    
    def add_marker(self, lat, lon, label="Marker"):
        """Add a single marker."""
        if not self.map_widget:
            return None
        
        obj_id = self.next_object_id
        self.next_object_id += 1
        
        try:
            marker_obj = self.map_widget.set_marker(
                lat, lon,
                text=label,
                marker_color_circle=self.current_color,
                marker_color_outside='gray'
            )
            
            self.drawn_objects[obj_id] = {
                'type': 'marker',
                'coords': [(lat, lon)],
                'color': self.current_color,
                'label': label,
                'objects': [marker_obj]
            }
            
            return obj_id
        except Exception as e:
            print(f"Error adding marker: {e}")
            return None
    
    def add_line_or_polygon(self, obj_type, points, label=""):
        """Add a line or polygon from a list of points."""
        if len(points) < 2 or not self.map_widget:
            return None
        
        obj_id = self.next_object_id
        self.next_object_id += 1
        
        try:
            if obj_type == "line":
                # Draw line segments
                line_objs = []
                for i in range(len(points) - 1):
                    try:
                        line = self.map_widget.set_path(
                            [points[i], points[i + 1]],
                            color=self.current_color,
                            width=2
                        )
                        line_objs.append(line)
                    except Exception as e:
                        print(f"Error adding line segment: {e}")
                        continue
                
                self.drawn_objects[obj_id] = {
                    'type': 'line',
                    'coords': points,
                    'color': self.current_color,
                    'label': label or f"Line {obj_id}",
                    'objects': line_objs
                }
            
            elif obj_type == "polygon":
                # Draw polygon (closed shape)
                polygon_points = points + [points[0]]  # Close the polygon
                line_objs = []
                for i in range(len(polygon_points) - 1):
                    try:
                        line = self.map_widget.set_path(
                            [polygon_points[i], polygon_points[i + 1]],
                            color=self.current_color,
                            width=2
                        )
                        line_objs.append(line)
                    except Exception as e:
                        print(f"Error adding polygon segment: {e}")
                        continue
                
                # Add corner markers for polygon
                for i, point in enumerate(points):
                    try:
                        marker = self.map_widget.set_marker(
                            point[0], point[1],
                            text=f"V{i+1}",
                            marker_color_circle=self.current_color,
                            marker_color_outside='gray'
                        )
                        line_objs.append(marker)
                    except Exception as e:
                        print(f"Error adding polygon marker: {e}")
                        continue
                
                self.drawn_objects[obj_id] = {
                    'type': 'polygon',
                    'coords': points,
                    'color': self.current_color,
                    'label': label or f"Polygon {obj_id}",
                    'objects': line_objs
                }
            
            return obj_id
        except Exception as e:
            print(f"Error adding line/polygon: {e}")
            return None
    
    def delete_object(self, obj_id):
        """Delete a drawn object by ID."""
        if obj_id in self.drawn_objects:
            obj_data = self.drawn_objects[obj_id]
            for obj in obj_data['objects']:
                try:
                    obj.delete()
                except:
                    pass
            del self.drawn_objects[obj_id]
            return True
        return False
    
    def edit_object(self, obj_id, new_label=None, new_color=None):
        """Edit an object's label and color."""
        if obj_id not in self.drawn_objects:
            return False
        
        obj_data = self.drawn_objects[obj_id]
        
        if new_label:
            obj_data['label'] = new_label
        
        if new_color and new_color in self.colors.values():
            obj_data['color'] = new_color
        
        return True
    
    def get_objects_at_location(self, lat, lon, tolerance=0.001):
        """Find drawn objects near a location (for click detection)."""
        nearby = []
        for obj_id, obj_data in self.drawn_objects.items():
            for coord in obj_data['coords']:
                if abs(coord[0] - lat) < tolerance and abs(coord[1] - lon) < tolerance:
                    nearby.append(obj_id)
                    break
        return nearby
    
    def clear_all(self):
        """Delete all drawn objects."""
        for obj_id in list(self.drawn_objects.keys()):
            self.delete_object(obj_id)

# --- Main App ---
class DroneAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Feed Analysis & Anomaly Detection")
        self.root.geometry("1400x950")
        self.root.configure(bg="#1e1e1e")

        # --- State Variables ---
        self.video_source = None
        self.cap = None
        self.is_playing = False
        self.thread = None
        self.stop_event = threading.Event()
        self.frame_width = 640
        self.frame_height = 360 
        self.current_frame_rgb = None # Store current frame for capture
        
        # Telemetry State (Static)
        self.telemetry_lat = 38.982
        self.telemetry_lon = -76.483
        self.telemetry_alt = 200 # feet
        self.telemetry_heading = 0
        
        # Selection State
        self.manual_roi_active = tk.BooleanVar(value=False)
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_id = None
        self.captured_images = [] # Keep refs to prevent GC
        self.captured_images_full = [] # Store full resolution images for tuning

        # Playback Control Vars
        self.playback_speed = tk.DoubleVar(value=1.0)
        self.pixel_lock_active = tk.BooleanVar(value=False)
        self.prev_gray = None 
        
        # Tracker Instance
        self.tracker_system = CentroidTracker(max_disappeared=20) 
        
        # CV Method Selection
        self.cv_method = tk.IntVar(value=5) 
        
        # Global Settings
        self.use_clahe = tk.BooleanVar(value=False)
        self.sobel_direction = tk.StringVar(value="Both")
        self.use_otsu = tk.BooleanVar(value=True) 
        self.enable_edge_filter = tk.BooleanVar(value=True)
        
        # MOG2 Subtractor instance
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        
        # Tune Tab Variables
        self.tune_image = None  # Current image being tuned (numpy array)
        self.tune_roi_bbox = None  # ROI bounding box (x, y, w, h)
        self.tune_canvas = None  # Canvas for displaying tuned image
        self.tune_photo = None  # PhotoImage reference
        self.master_sensitivity = tk.DoubleVar(value=1.0)  # Master sensitivity scalar
        self.tune_last_render_rgb = None
        self.tune_last_detected_rects = []
        self.tune_draw_targets = tk.BooleanVar(value=False)
        self.tune_gt_mode = tk.StringVar(value="Box")
        self.tune_target_count = tk.IntVar(value=1)
        self.tune_calibration_status = tk.StringVar(value="Idle")
        self.tune_display_transform = {
            'scale': 1.0,
            'offset_x': 0,
            'offset_y': 0,
            'draw_w': 0,
            'draw_h': 0,
            'img_w': 0,
            'img_h': 0
        }
        self.tune_gt_annotations = {'boxes': [], 'polygons': [], 'points': []}
        self.tune_gt_polygon_working = []
        self.tune_gt_temp_box = None
        self.tune_gt_drag_start = None
        self.tune_current_image_key = None
        self.tune_gt_cache = {}
        self.tune_is_calibrating = False
        self.tune_calibration_queue = None
        self.tune_calibration_thread = None
        self.tune_skip_slider_callback = False
        
        # Tune Tab Sliders (separate from main sliders)
        self.tune_slider_clahe_clip = None
        self.tune_slider_clahe_grid = None
        self.tune_slider_canny_min = None
        self.tune_slider_canny_max = None
        self.tune_slider_gauss = None
        self.tune_slider_blob_min_thresh = None
        self.tune_slider_blob_max_thresh = None
        self.tune_slider_blob_min_area = None
        self.tune_slider_edge_density = None
        self.tune_use_otsu = tk.BooleanVar(value=True)
        self.tune_enable_edge_filter = tk.BooleanVar(value=True)
        
        # Map-related state
        self.map_widget = None
        self.marker_manager = None
        self.drawing_tools = None
        self.topo_overlay = None
        self.layer_filters = {
            'Hostile': tk.BooleanVar(value=True),
            'Friendly': tk.BooleanVar(value=True),
            'Neutral': tk.BooleanVar(value=True),
            'Unknown': tk.BooleanVar(value=True),
            'Civilian': tk.BooleanVar(value=True)
        }
        
        # TIFF overlay state
        self.tiff_canvas = None
        self.tiff_image = None
        self.tiff_photo_image = None
        self.tiff_path = None
        self.tiff_zoom = 1.0
        self.tiff_pan_x = 0
        self.tiff_pan_y = 0
        self.tiff_zoom_cache = {}  # Cache resized images
        self.tiff_drawing_tools = None  # For TIFF annotations
        self.tiff_pan_start_x = 0
        self.tiff_pan_start_y = 0

        # --- Styles ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#1e1e1e", foreground="white", font=("Arial", 10))
        style.configure("TButton", background="#333333", foreground="white", borderwidth=1)
        style.configure("TFrame", background="#1e1e1e")
        style.configure("Header.TLabel", font=("Arial", 12, "bold"), foreground="#00ff00")
        style.configure("Section.TLabelframe", background="#1e1e1e", foreground="#00ff00", bordercolor="gray")
        style.configure("Section.TLabelframe.Label", background="#1e1e1e", foreground="#00ff00")
        
        # Custom Scrollbar style
        style.configure("Vertical.TScrollbar", background="#333", troughcolor="#1e1e1e", bordercolor="#333", arrowcolor="white")

        # --- Scrollable Setup ---
        main_container = tk.Frame(self.root, bg="#1e1e1e")
        main_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main_container, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.scrollable_frame = tk.Frame(self.canvas, bg="#1e1e1e")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel) 
        self.root.bind_all("<Button-4>", self._on_mousewheel)
        self.root.bind_all("<Button-5>", self._on_mousewheel)

        # --- GUI Layout ---
        self.create_layout()
        self.update_settings_visibility()
        self.cv_method.trace_add("write", self.update_settings_visibility)
        self.update_drone_telemetry()

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        if event.num == 5 or event.delta == -120:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta == 120:
            self.canvas.yview_scroll(-1, "units")

    def create_layout(self):
        # Configure grid weights for proper resizing
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=0)
        self.scrollable_frame.grid_columnconfigure(2, weight=2)
        self.scrollable_frame.grid_rowconfigure(0, weight=1)
        
        # === COLUMN 0: VIDEO FEEDS & CONTROLS ===
        self.frame_video = tk.Frame(self.scrollable_frame, bg="#1e1e1e", padx=10, pady=10)
        self.frame_video.grid(row=0, column=0, sticky="nsew")

        tk.Label(self.frame_video, text="Raw Drone Video Feed", bg="black", fg="white", font=("Arial", 12, "bold")).pack(fill="x")
        self.canvas_raw = tk.Canvas(self.frame_video, width=self.frame_width, height=self.frame_height, bg="black", highlightthickness=0)
        self.canvas_raw.pack(pady=(0, 10))
        
        # Bind Mouse Events for ROI Selection
        self.canvas_raw.bind("<ButtonPress-1>", self.on_roi_start)
        self.canvas_raw.bind("<B1-Motion>", self.on_roi_drag)
        self.canvas_raw.bind("<ButtonRelease-1>", self.on_roi_end)

        tk.Label(self.frame_video, text="Drone Video Feed Overlay Detection", bg="black", fg="white", font=("Arial", 12, "bold")).pack(fill="x")
        self.canvas_processed = tk.Canvas(self.frame_video, width=self.frame_width, height=self.frame_height, bg="black", highlightthickness=0)
        self.canvas_processed.pack()

        # Playback Controls
        control_frame = ttk.LabelFrame(self.frame_video, text="Playback Controls", style="Section.TLabelframe", padding=5)
        control_frame.pack(fill="x", pady=10)

        self.btn_load = tk.Button(control_frame, text="📁 Load Video File", command=self.load_video, 
                                 bg="#444", fg="white", font=("Arial", 9, "bold"), 
                                 height=2, relief="raised", bd=2)
        self.btn_load.pack(fill="x", pady=2)
        
        # Video Playback Control Buttons
        playback_btn_frame = tk.Frame(control_frame, bg="#1e1e1e")
        playback_btn_frame.pack(fill="x", pady=5)
        
        playback_btn_frame.grid_columnconfigure(0, weight=1)
        playback_btn_frame.grid_columnconfigure(1, weight=1)
        playback_btn_frame.grid_columnconfigure(2, weight=1)
        
        self.btn_rewind = tk.Button(playback_btn_frame, text="⏪ Rewind", command=self.rewind_video,
                                    bg="#f0ad4e", fg="white", font=("Arial", 9, "bold"),
                                    height=1, relief="raised", bd=2, state="disabled")
        self.btn_rewind.grid(row=0, column=0, sticky="ew", padx=2)
        
        self.btn_play_pause = tk.Button(playback_btn_frame, text="▶ Play", command=self.toggle_play_pause,
                                        bg="#5cb85c", fg="white", font=("Arial", 9, "bold"),
                                        height=1, relief="raised", bd=2, state="disabled")
        self.btn_play_pause.grid(row=0, column=1, sticky="ew", padx=2)
        
        self.btn_stop = tk.Button(playback_btn_frame, text="⏹ Stop", command=self.stop_video,
                                 bg="#d9534f", fg="white", font=("Arial", 9, "bold"),
                                 height=1, relief="raised", bd=2, state="disabled")
        self.btn_stop.grid(row=0, column=2, sticky="ew", padx=2)

        # -- New ROI Selection Toggle --
        roi_frame = tk.Frame(control_frame, bg="#1e1e1e")
        roi_frame.pack(fill="x", pady=5)
        self.chk_roi = tk.Checkbutton(roi_frame, text="Manual ROI Selection", variable=self.manual_roi_active, 
                                      bg="#1e1e1e", fg="#00ff00", selectcolor="#444", 
                                      activebackground="#1e1e1e", activeforeground="#00ff00",
                                      font=("Arial", 9, "bold"))
        self.chk_roi.pack(side="left")
        tk.Label(roi_frame, text="(Drag on Raw Feed)", bg="#1e1e1e", fg="gray", 
                font=("Arial", 8, "italic")).pack(side="left", padx=3)

        tk.Label(control_frame, text="Playback Speed", bg="#1e1e1e", fg="white", font=("Arial", 8)).pack(anchor="w")
        self.speed_slider = tk.Scale(control_frame, from_=0.1, to=2.0, resolution=0.1, orient="horizontal", 
                                     variable=self.playback_speed, bg="#1e1e1e", fg="white", troughcolor="#444", bd=0, highlightthickness=0)
        self.speed_slider.set(1.0)
        self.speed_slider.pack(fill="x")

        self.chk_lock = tk.Checkbutton(control_frame, text="Pixel Lock (Stabilize)", variable=self.pixel_lock_active, 
                                       bg="#1e1e1e", fg="#00ff00", selectcolor="#444", 
                                       activebackground="#1e1e1e", activeforeground="#00ff00",
                                       font=("Arial", 9, "bold"))
        self.chk_lock.pack(fill="x", pady=5)
        
        tk.Button(control_frame, text="🔄 Reset Active Locks", command=self.reset_trackers, 
                 bg="#d9534f", fg="white", font=("Arial", 9, "bold"),
                 height=1, relief="raised", bd=2).pack(fill="x", pady=2)


        # === COLUMN 1: DRONE DATA ===
        self.frame_data = tk.Frame(self.scrollable_frame, bg="#1e1e1e", padx=10, pady=10, width=300)
        self.frame_data.grid(row=0, column=1, sticky="ns")

        data_group = ttk.LabelFrame(self.frame_data, text="Drone Flight Data & Status", style="Section.TLabelframe", padding=10)
        data_group.pack(fill="x", pady=5)

        self.lbl_altitude = ttk.Label(data_group, text="Altitude: -- m")
        self.lbl_altitude.pack(anchor="w")
        self.lbl_gps = ttk.Label(data_group, text="GPS: --, --")
        self.lbl_gps.pack(anchor="w")
        self.lbl_grid = ttk.Label(data_group, text="Grid Coord: --")
        self.lbl_grid.pack(anchor="w")
        self.lbl_time = ttk.Label(data_group, text="Flight Time: -- min")
        self.lbl_time.pack(anchor="w")
        
        ttk.Separator(data_group, orient='horizontal').pack(fill='x', pady=5)
        
        self.lbl_orient = ttk.Label(data_group, text="Orientation: Top Down")
        self.lbl_orient.pack(anchor="w")
        self.lbl_angle = ttk.Label(data_group, text="Camera Angle: 45°")
        self.lbl_angle.pack(anchor="w")
        self.lbl_zoom = ttk.Label(data_group, text="Zoom: 1.0x")
        self.lbl_zoom.pack(anchor="w")

        btn_anomaly = tk.Button(data_group, text="⚠ Record Anomaly Location", bg="#d9534f", fg="white",
                               font=("Arial", 9, "bold"), height=2, relief="raised", bd=2)
        btn_anomaly.pack(fill="x", pady=10)

        # CV Method Selector
        method_group = ttk.LabelFrame(self.frame_data, text="Active CV Method", style="Section.TLabelframe", padding=10)
        method_group.pack(fill="x", pady=10)
        
        tk.Radiobutton(method_group, text="Combined (Detection)", variable=self.cv_method, value=5, bg="#1e1e1e", fg="white", selectcolor="#444").pack(anchor="w")
        tk.Radiobutton(method_group, text="Edge Detection (Canny)", variable=self.cv_method, value=2, bg="#1e1e1e", fg="white", selectcolor="#444").pack(anchor="w")
        tk.Radiobutton(method_group, text="Motion (Standard)", variable=self.cv_method, value=3, bg="#1e1e1e", fg="white", selectcolor="#444").pack(anchor="w")
        tk.Radiobutton(method_group, text="Sobel Filter", variable=self.cv_method, value=4, bg="#1e1e1e", fg="white", selectcolor="#444").pack(anchor="w")

        # === COLUMN 2: TABBED INTERFACE (Settings & Detections) ===
        self.right_panel = tk.Frame(self.scrollable_frame, bg="#1e1e1e", padx=10, pady=10)
        self.right_panel.grid(row=0, column=2, sticky="nsew")
        
        # Configure Notebook style
        style = ttk.Style() 
        style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#333", foreground="black", padding=[10, 5], font=("Arial", 10))
        style.map("TNotebook.Tab", background=[("selected", "#00ff00")], foreground=[("selected", "black")])

        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill="both", expand=True)
        
        # Bind tab change event to show/hide video feeds
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Tab 1: CV Settings
        self.tab_settings = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_settings, text="CV Settings")
        
        # Tab 2: Saved Detections
        self.tab_detections = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_detections, text="Saved Detections")
        
        # Tab 3: Topographic Map
        self.tab_map = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_map, text="Topographic Map")
        
        # Tab 4: Uploaded TIFF
        self.tab_tiff = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_tiff, text="Uploaded TIFF")
        
        # Tab 5: Tune
        self.tab_tune = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab_tune, text="Tune")

        # --- SETUP SETTINGS TAB (Moved from old frame_settings) ---
        # Set frame_settings alias to tab_settings so existing code works
        self.frame_settings = self.tab_settings

        # -- GLOBAL SETTINGS (CLAHE) --
        self.global_frame = ttk.LabelFrame(self.frame_settings, text="Image Enhancement (Global)", style="Section.TLabelframe", padding=5)
        self.global_frame.pack(fill="x", pady=5)
        
        tk.Checkbutton(self.global_frame, text="Enable Histogram Eq (CLAHE)", variable=self.use_clahe,
                       bg="#1e1e1e", fg="#00ff00", selectcolor="#444", activebackground="#1e1e1e", activeforeground="#00ff00").pack(anchor="w", pady=2)
        self.slider_clahe_clip = self.create_slider(self.global_frame, "Clip Limit (Contrast)", 1, 10, 3)
        self.slider_clahe_grid = self.create_slider(self.global_frame, "Grid Size (Locality)", 1, 16, 8)

        # -- Blob Sliders Frame --
        self.blob_frame = ttk.LabelFrame(self.frame_settings, text="Hot Blob Threshold Sliders", style="Section.TLabelframe", padding=5)
        tk.Checkbutton(self.blob_frame, text="Use Otsu Thresholding", variable=self.use_otsu,
                       bg="#1e1e1e", fg="#00ff00", selectcolor="#444", activebackground="#1e1e1e", activeforeground="#00ff00").pack(anchor="w", pady=2)
        self.slider_blob_min_thresh = self.create_slider(self.blob_frame, "Min Threshold", 10, 255, 120) 
        self.slider_blob_max_thresh = self.create_slider(self.blob_frame, "Max Threshold", 10, 255, 255)
        self.slider_blob_min_area = self.create_slider(self.blob_frame, "Min Area", 10, 500, 25) 

        # -- Edge Detection Sliders Frame --
        self.edge_frame = ttk.LabelFrame(self.frame_settings, text="Edge Detection Sliders (Canny)", style="Section.TLabelframe", padding=5)
        self.slider_canny_min = self.create_slider(self.edge_frame, "Min Gradient", 0, 255, 20) 
        self.slider_canny_max = self.create_slider(self.edge_frame, "Max Gradient", 0, 255, 80) 
        self.slider_gauss = self.create_slider(self.edge_frame, "Gaussian Blur", 1, 15, 3)

        # -- Combined Mode Specifics Frame --
        self.combined_frame = ttk.LabelFrame(self.frame_settings, text="Combined Mode Settings", style="Section.TLabelframe", padding=5)
        tk.Checkbutton(self.combined_frame, text="Enable Edge Filtering", variable=self.enable_edge_filter,
                       bg="#1e1e1e", fg="#00ff00", selectcolor="#444", activebackground="#1e1e1e", activeforeground="#00ff00").pack(anchor="w", pady=2)
        self.slider_edge_density = self.create_slider(self.combined_frame, "Edge Density Threshold", 1, 100, 3) 

        # -- Sobel Sliders Frame --
        self.sobel_frame = ttk.LabelFrame(self.frame_settings, text="Sobel Filter Settings", style="Section.TLabelframe", padding=5)
        tk.Label(self.sobel_frame, text="Filter Direction", bg="#1e1e1e", fg="#aaaaaa", font=("Arial", 8)).pack(anchor="w")
        self.cmb_sobel_dir = ttk.Combobox(self.sobel_frame, textvariable=self.sobel_direction, values=["Horizontal", "Vertical", "Both"], state="readonly")
        self.cmb_sobel_dir.pack(fill="x", pady=(0, 5))
        self.slider_sobel_k = self.create_slider(self.sobel_frame, "Kernel Size (Odd)", 1, 7, 3)
        self.slider_sobel_weight = self.create_slider(self.sobel_frame, "Overlay Intensity", 1, 10, 5)

        # -- MOG2 Motion Sliders Frame --
        self.mog_frame = ttk.LabelFrame(self.frame_settings, text="MOG2 Motion Detection", style="Section.TLabelframe", padding=5)
        self.slider_mog_history = self.create_slider(self.mog_frame, "History Length", 10, 500, 200)
        self.slider_mog_var = self.create_slider(self.mog_frame, "Variance Threshold", 1, 100, 16)

        # -- YOLO Placeholder Frame --
        self.yolo_frame = ttk.LabelFrame(self.frame_settings, text="YOLO Detection Settings (Inactive)", style="Section.TLabelframe", padding=5)
        self.create_slider(self.yolo_frame, "Detection Confidence Level", 0, 100, 60)
        yolo_grid = tk.Frame(self.yolo_frame, bg="#1e1e1e")
        yolo_grid.pack(pady=5)
        labels = [("Vehicles", "#5cb85c"), ("People", "#0275d8"), ("LSC", "#f0ad4e"), ("Drones", "#d9534f"), 
                  ("Tree-line", "#5cb85c"), ("Anomaly", "#292b2c"), ("Weapon", "#8e44ad"), ("Civilian", "#800000"), ("Trench", "#5bc0de")]
        for i, (text, color) in enumerate(labels):
            r = i // 3; c = i % 3
            lbl = tk.Label(yolo_grid, text=text, bg=color, fg="white", width=8, height=2, font=("Arial", 8, "bold"))
            lbl.grid(row=r, column=c, padx=2, pady=2)
            tk.Checkbutton(yolo_grid, bg=color, activebackground=color).grid(row=r, column=c, sticky="se")

        # --- SETUP DETECTIONS TAB ---
        # Controls for detections
        det_controls = tk.Frame(self.tab_detections, bg="#1e1e1e")
        det_controls.pack(fill="x", pady=5, padx=5)
        
        # Info label
        tk.Label(det_controls, text="💡 Tip: Click '🔧 Tune This' on any detection to fine-tune CV parameters in the Tune tab",
                bg="#1e1e1e", fg="#00ff00", font=("Arial", 8), wraplength=350, justify="left").pack(anchor="w", pady=(0, 5))
        
        tk.Button(det_controls, text="🗑 Clear All Detections", command=self.clear_detections, 
                 bg="#d9534f", fg="white", font=("Arial", 9, "bold"), 
                 height=2, relief="raised", bd=2).pack(fill="x")
        
        # Scrollable area for thumbnails
        self.det_canvas = tk.Canvas(self.tab_detections, bg="#1e1e1e", highlightthickness=0)
        self.det_scrollbar = ttk.Scrollbar(self.tab_detections, orient="vertical", command=self.det_canvas.yview)
        self.det_frame = tk.Frame(self.det_canvas, bg="#1e1e1e")
        
        self.det_frame.bind("<Configure>", lambda e: self.det_canvas.configure(scrollregion=self.det_canvas.bbox("all")))
        self.det_window = self.det_canvas.create_window((0, 0), window=self.det_frame, anchor="nw")
        
        self.det_canvas.configure(yscrollcommand=self.det_scrollbar.set)
        self.det_canvas.pack(side="left", fill="both", expand=True)
        self.det_scrollbar.pack(side="right", fill="y")
        
        # Ensure inner frame expands to canvas width
        self.det_canvas.bind("<Configure>", lambda e: self.det_canvas.itemconfig(self.det_window, width=e.width))
        
        # --- SETUP MAP TAB ---
        self.setup_map_tab()
        
        # --- SETUP TIFF TAB ---
        self.setup_tiff_tab()
        
        # --- SETUP TUNE TAB ---
        self.setup_tune_tab()
        
        # Initially show video feeds (CV Settings is first tab)
        self.on_tab_changed(None)

    def on_tab_changed(self, event):
        """Handle tab change to show/hide video feeds."""
        try:
            current_tab = self.notebook.index(self.notebook.select())
            
            # Tab 0 is CV Settings - show video feeds
            if current_tab == 0:
                self.frame_video.grid(row=0, column=0, sticky="nsew")
                self.frame_data.grid(row=0, column=1, sticky="ns")
                self.right_panel.grid(row=0, column=2, sticky="nsew")
            else:
                # Hide video feeds for all other tabs and expand right panel
                self.frame_video.grid_forget()
                self.frame_data.grid_forget()
                # Make right panel span all columns
                self.right_panel.grid(row=0, column=0, columnspan=3, sticky="nsew")
        except Exception as e:
            print(f"Error in tab change handler: {e}")

    def setup_map_tab(self):
        """Initialize the topographic map tab with controls and map widget."""
        # Top control panel
        control_panel = tk.Frame(self.tab_map, bg="#1e1e1e", height=60)
        control_panel.pack(fill="x", padx=5, pady=5)
        
        # Button row
        btn_frame = tk.Frame(control_panel, bg="#1e1e1e")
        btn_frame.pack(fill="x", pady=(0, 5))
        
        tk.Button(btn_frame, text="🎯 Center on Detections", command=self.center_on_detections, 
                 bg="#0275d8", fg="white", font=("Arial", 8, "bold"), width=16).pack(side="left", padx=2)
        tk.Button(btn_frame, text="🚫 Clear Markers", command=self.clear_all_markers, 
                 bg="#d9534f", fg="white", font=("Arial", 8, "bold"), width=13).pack(side="left", padx=2)
        tk.Button(btn_frame, text="🗑 Clear Drawings", command=self.clear_all_drawings, 
                 bg="#d9534f", fg="white", font=("Arial", 8, "bold"), width=13).pack(side="left", padx=2)
        tk.Button(btn_frame, text="🔄 Reset View", command=self.reset_map_view, 
                 bg="#f0ad4e", fg="white", font=("Arial", 8, "bold"), width=12).pack(side="left", padx=2)
        
        # Drawing tools row
        drawing_frame = tk.Frame(control_panel, bg="#1e1e1e")
        drawing_frame.pack(fill="x", pady=(0, 5))
        
        tk.Label(drawing_frame, text="Drawing Tools:", bg="#1e1e1e", fg="white", font=("Arial", 9)).pack(side="left", padx=(0, 5))
        self.drawing_mode_var = tk.StringVar(value="None")
        drawing_combo = ttk.Combobox(drawing_frame, textvariable=self.drawing_mode_var, 
                                     values=["None", "Marker", "Line", "Polygon"], 
                                     state="readonly", width=10)
        drawing_combo.pack(side="left", padx=2)
        drawing_combo.bind("<<ComboboxSelected>>", self.on_drawing_mode_changed)
        
        tk.Label(drawing_frame, text="Color:", bg="#1e1e1e", fg="white", font=("Arial", 9)).pack(side="left", padx=(5, 0))
        self.color_var = tk.StringVar(value="Red")
        color_combo = ttk.Combobox(drawing_frame, textvariable=self.color_var, 
                                   values=["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange", "Purple", "White"], 
                                   state="readonly", width=10)
        color_combo.pack(side="left", padx=2)
        color_combo.bind("<<ComboboxSelected>>", self.on_color_changed)
        
        self.finish_drawing_btn = tk.Button(drawing_frame, text="✓ Finish", command=self.finish_drawing, 
                                            bg="#f0ad4e", fg="white", font=("Arial", 8, "bold"), 
                                            state="disabled", width=10)
        self.finish_drawing_btn.pack(side="left", padx=2)
        
        # Tile server selection
        tile_frame = tk.Frame(control_panel, bg="#1e1e1e")
        tile_frame.pack(fill="x")
        tk.Label(tile_frame, text="Map Type:", bg="#1e1e1e", fg="white", font=("Arial", 9)).pack(side="left", padx=(0, 5))
        self.tile_server_var = tk.StringVar(value="OpenStreetMap")
        tile_combo = ttk.Combobox(tile_frame, textvariable=self.tile_server_var, 
                                  values=["OpenStreetMap", "Google Satellite", "OpenTopoMap"], 
                                  state="readonly", width=20)
        tile_combo.pack(side="left")
        tile_combo.bind("<<ComboboxSelected>>", self.change_tile_server)
        
        # Status label
        self.map_status_label = tk.Label(tile_frame, text="Map Status: Ready", 
                                         bg="#1e1e1e", fg="#00ff00", font=("Arial", 8))
        self.map_status_label.pack(side="right", padx=5)
        
        # Map widget
        map_container = tk.Frame(self.tab_map, bg="#1e1e1e")
        map_container.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        try:
            self.map_widget = tkintermapview.TkinterMapView(
                map_container, 
                width=800, 
                height=500, 
                corner_radius=0
            )
            self.map_widget.pack(fill="both", expand=True)
            
            # Set initial position to NSA Annapolis
            self.map_widget.set_position(self.telemetry_lat, self.telemetry_lon)
            self.map_widget.set_zoom(15)
            
            # Initialize marker manager and drawing tools
            self.marker_manager = MapMarkerManager(self.map_widget, self.telemetry_lat, self.telemetry_lon)
            self.drawing_tools = DrawingToolsManager(self.map_widget)
            
            # Add click handler for coordinates display and drawing
            self.map_widget.add_left_click_map_command(self.on_map_click_handler)
            
        except Exception as e:
            error_label = tk.Label(map_container, text=f"Error loading map: {e}", 
                                  bg="#1e1e1e", fg="red", font=("Arial", 10))
            error_label.pack(expand=True)
            print(f"Map initialization error: {e}")
        
        # Bottom control panel - Layer filters
        filter_panel = ttk.LabelFrame(self.tab_map, text="Layer Visibility", 
                                      style="Section.TLabelframe", padding=5)
        filter_panel.pack(fill="x", padx=5, pady=(0, 5))
        
        filter_row = tk.Frame(filter_panel, bg="#1e1e1e")
        filter_row.pack(fill="x")
        
        for affiliation in ['Hostile', 'Friendly', 'Neutral', 'Unknown', 'Civilian']:
            var = self.layer_filters[affiliation]
            chk = tk.Checkbutton(filter_row, text=f"Show {affiliation}", variable=var,
                               bg="#1e1e1e", fg="white", selectcolor="#444",
                               activebackground="#1e1e1e", activeforeground="#00ff00",
                               command=lambda a=affiliation: self.toggle_layer_visibility(a))
            chk.pack(side="left", padx=5)
        
        # Marker count display
        self.marker_count_label = tk.Label(filter_panel, text="Markers: 0", 
                                          bg="#1e1e1e", fg="#00ff00", font=("Arial", 9, "bold"))
        self.marker_count_label.pack(pady=(5, 0))
    
    def on_drawing_mode_changed(self, event=None):
        """Handle drawing mode change."""
        if not self.drawing_tools:
            return
        
        mode = self.drawing_mode_var.get()
        self.drawing_tools.set_drawing_mode(mode)
        
        if mode in ["Line", "Polygon"]:
            self.finish_drawing_btn.config(state="normal")
            self.map_status_label.config(text=f"Click map to add points. Press 'Finish {mode}' when done.")
        else:
            self.finish_drawing_btn.config(state="disabled")
            if mode == "Marker":
                self.map_status_label.config(text="Click map to place markers")
            else:
                self.map_status_label.config(text="Ready")
    
    def on_color_changed(self, event=None):
        """Handle color change."""
        if not self.drawing_tools:
            return
        
        color_name = self.color_var.get()
        self.drawing_tools.set_color(color_name)
        self.map_status_label.config(text=f"Color changed to {color_name}")
    
    def finish_drawing(self):
        """Finish drawing a line or polygon."""
        try:
            if not self.drawing_tools:
                self.map_status_label.config(text="Error: Drawing tools not initialized")
                return
            
            if self.drawing_tools.finish_line_or_polygon():
                self.map_status_label.config(text="Line/Polygon completed")
                # Reset drawing mode
                self.drawing_mode_var.set("None")
                self.on_drawing_mode_changed()
            else:
                self.map_status_label.config(text="Need at least 2 points to finish")
        except Exception as e:
            print(f"Error finishing drawing: {e}")
            self.map_status_label.config(text=f"Drawing error: {e}")
    
    def on_map_click_handler(self, coords):
        """Handle map clicks for both status display and drawing."""
        try:
            lat, lon = coords
            
            # Check if drawing tools should handle it
            if self.drawing_tools and self.drawing_tools.drawing_mode != "none":
                if self.drawing_tools.on_map_click(coords):
                    return
            
            # Check if clicking on a drawn object to edit it
            if self.drawing_tools:
                try:
                    nearby_objs = self.drawing_tools.get_objects_at_location(lat, lon)
                    if nearby_objs:
                        obj_id = nearby_objs[0]
                        self.show_drawing_edit_dialog(obj_id)
                        return
                except Exception as e:
                    print(f"Error checking for nearby objects: {e}")
            
            # Default: Show coordinates
            self.map_status_label.config(text=f"Clicked: {lat:.6f}, {lon:.6f}")
        except Exception as e:
            print(f"Error in on_map_click_handler: {e}")
            self.map_status_label.config(text=f"Click error: {e}")
    
    def show_drawing_edit_dialog(self, obj_id):
        """Show edit dialog for a drawn object."""
        try:
            if not self.drawing_tools or obj_id not in self.drawing_tools.drawn_objects:
                return
            
            obj_data = self.drawing_tools.drawn_objects[obj_id]
            
            # Create dialog window
            dialog = tk.Toplevel(self.root)
            dialog.title(f"Edit {obj_data['type'].upper()}")
            dialog.geometry("300x250")
            dialog.configure(bg="#1e1e1e")
            dialog.resizable(False, False)
            
            # Make it modal
            dialog.transient(self.root)
            dialog.grab_set()
            
            # --- Object Info ---
            info_frame = ttk.LabelFrame(dialog, text="Object Info", style="Section.TLabelframe", padding=10)
            info_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Label(info_frame, text=f"Type: {obj_data['type'].upper()}", 
                    bg="#1e1e1e", fg="white", font=("Arial", 10)).pack(anchor="w")
            
            # --- Label Editor ---
            label_frame = tk.Frame(dialog, bg="#1e1e1e")
            label_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Label(label_frame, text="Label:", bg="#1e1e1e", fg="white").pack(side="left", padx=(0, 5))
            label_entry = tk.Entry(label_frame, bg="#333", fg="white", width=20)
            label_entry.insert(0, obj_data['label'])
            label_entry.pack(side="left")
            
            # --- Color Picker ---
            color_frame = tk.Frame(dialog, bg="#1e1e1e")
            color_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Label(color_frame, text="Color:", bg="#1e1e1e", fg="white").pack(side="left", padx=(0, 5))
            color_combo = ttk.Combobox(color_frame, 
                                       values=["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange", "Purple", "White"],
                                       state="readonly", width=15)
            
            # Set current color
            for color_name, color_hex in self.drawing_tools.colors.items():
                if color_hex == obj_data['color']:
                    color_combo.set(color_name)
                    break
            
            color_combo.pack(side="left")
            
            # --- Buttons ---
            button_frame = tk.Frame(dialog, bg="#1e1e1e")
            button_frame.pack(fill="x", padx=10, pady=10)
            
            def save_changes():
                try:
                    new_label = label_entry.get().strip()
                    new_color = self.drawing_tools.colors.get(color_combo.get())
                    
                    if new_label:
                        self.drawing_tools.edit_object(obj_id, new_label, new_color)
                        self.map_status_label.config(text=f"Updated {obj_data['type']}")
                    
                    dialog.destroy()
                except Exception as e:
                    print(f"Error saving changes: {e}")
                    messagebox.showerror("Error", f"Error saving: {e}")
            
            def delete_object():
                try:
                    if messagebox.askyesno("Confirm Delete", 
                                           f"Are you sure you want to delete this {obj_data['type']}?"):
                        self.drawing_tools.delete_object(obj_id)
                        self.map_status_label.config(text=f"{obj_data['type']} deleted")
                        dialog.destroy()
                except Exception as e:
                    print(f"Error deleting object: {e}")
                    messagebox.showerror("Error", f"Error deleting: {e}")
            
            tk.Button(button_frame, text="Save", command=save_changes, 
                     bg="#5cb85c", fg="white", font=("Arial", 9, "bold")).pack(side="left", padx=2, fill="x", expand=True)
            tk.Button(button_frame, text="Delete", command=delete_object, 
                     bg="#d9534f", fg="white", font=("Arial", 9, "bold")).pack(side="left", padx=2, fill="x", expand=True)
            tk.Button(button_frame, text="Cancel", command=dialog.destroy, 
                     bg="#666", fg="white", font=("Arial", 9, "bold")).pack(side="left", padx=2, fill="x", expand=True)
        except Exception as e:
            print(f"Error showing edit dialog: {e}")
            messagebox.showerror("Error", f"Error opening edit dialog: {e}")
    
    def center_on_topo(self):
        """Center map on the uploaded topographic file bounds."""
        if not self.map_widget or not self.topo_overlay:
            messagebox.showinfo("No Topo File", "Please upload a topographic map file first.")
            return
        
        bounds = self.topo_overlay['bounds']
        try:
            self.map_widget.fit_bounding_box(bounds[0], bounds[1])
            self.map_status_label.config(text="Centered on topographic map")
        except Exception as e:
            self.map_status_label.config(text=f"Error centering on topo: {e}")
    
    def setup_tiff_tab(self):
        """Initialize the TIFF image viewer tab with drawing tools."""
        # Top control panel
        control_panel = tk.Frame(self.tab_tiff, bg="#1e1e1e")
        control_panel.pack(fill="x", padx=5, pady=5)
        
        # Button row
        btn_frame = tk.Frame(control_panel, bg="#1e1e1e")
        btn_frame.pack(fill="x", pady=(0, 5))
        
        tk.Button(btn_frame, text="📁 Upload TIFF", command=self.upload_tiff_file, 
                 bg="#5cb85c", fg="white", font=("Arial", 8, "bold"), width=12).pack(side="left", padx=2)
        tk.Button(btn_frame, text="⛶ Fit Window", command=self.fit_tiff_to_window, 
                 bg="#0275d8", fg="white", font=("Arial", 8, "bold"), width=11).pack(side="left", padx=2)
        tk.Button(btn_frame, text="🔍+ Zoom In", command=self.zoom_tiff_in, 
                 bg="#f0ad4e", fg="white", font=("Arial", 8, "bold"), width=10).pack(side="left", padx=2)
        tk.Button(btn_frame, text="🔍- Zoom Out", command=self.zoom_tiff_out, 
                 bg="#f0ad4e", fg="white", font=("Arial", 8, "bold"), width=11).pack(side="left", padx=2)
        tk.Button(btn_frame, text="🔄 Reset", command=self.reset_tiff_view, 
                 bg="#666", fg="white", font=("Arial", 8, "bold"), width=9).pack(side="left", padx=2)
        tk.Button(btn_frame, text="🗑 Clear", command=self.clear_tiff_drawings, 
                 bg="#d9534f", fg="white", font=("Arial", 8, "bold"), width=9).pack(side="left", padx=2)
        
        # Drawing tools row
        drawing_frame = tk.Frame(control_panel, bg="#1e1e1e")
        drawing_frame.pack(fill="x", pady=(0, 5))
        
        tk.Label(drawing_frame, text="Drawing Tools:", bg="#1e1e1e", fg="white", font=("Arial", 9)).pack(side="left", padx=(0, 5))
        self.tiff_drawing_mode_var = tk.StringVar(value="None")
        tiff_drawing_combo = ttk.Combobox(drawing_frame, textvariable=self.tiff_drawing_mode_var, 
                                         values=["None", "Marker", "Line", "Polygon"], 
                                         state="readonly", width=10)
        tiff_drawing_combo.pack(side="left", padx=2)
        tiff_drawing_combo.bind("<<ComboboxSelected>>", self.on_tiff_drawing_mode_changed)
        
        tk.Label(drawing_frame, text="Color:", bg="#1e1e1e", fg="white", font=("Arial", 9)).pack(side="left", padx=(5, 0))
        self.tiff_color_var = tk.StringVar(value="Red")
        tiff_color_combo = ttk.Combobox(drawing_frame, textvariable=self.tiff_color_var, 
                                       values=["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange", "Purple", "White"], 
                                       state="readonly", width=10)
        tiff_color_combo.pack(side="left", padx=2)
        tiff_color_combo.bind("<<ComboboxSelected>>", self.on_tiff_color_changed)
        
        self.tiff_finish_drawing_btn = tk.Button(drawing_frame, text="Finish Line/Polygon", command=self.finish_tiff_drawing, 
                                                 bg="#f0ad4e", fg="white", font=("Arial", 8, "bold"), state="disabled")
        self.tiff_finish_drawing_btn.pack(side="left", padx=2)
        
        # Status label
        self.tiff_status_label = tk.Label(control_panel, text="No TIFF loaded", 
                                         bg="#1e1e1e", fg="#ffff00", font=("Arial", 8))
        self.tiff_status_label.pack(side="right", padx=5)
        
        # Canvas for TIFF display
        canvas_frame = tk.Frame(self.tab_tiff, bg="#1e1e1e")
        canvas_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        
        self.tiff_canvas = tk.Canvas(canvas_frame, bg="black", highlightthickness=1, highlightbackground="gray")
        self.tiff_canvas.pack(fill="both", expand=True)
        
        # Bind mouse events - optimized for panning and drawing
        self.tiff_canvas.bind("<MouseWheel>", self.tiff_mousewheel)
        self.tiff_canvas.bind("<Button-4>", self.tiff_mousewheel)
        self.tiff_canvas.bind("<Button-5>", self.tiff_mousewheel)
        self.tiff_canvas.bind("<B3-Motion>", self.tiff_pan)  # Right-click drag to pan
        self.tiff_canvas.bind("<ButtonPress-3>", self.tiff_pan_start)
        self.tiff_canvas.bind("<B1-Motion>", self.tiff_pan)  # Left-click drag  to pan (alt)
        self.tiff_canvas.bind("<ButtonPress-1>", self.tiff_pan_start)
        self.tiff_canvas.bind("<ButtonRelease-1>", self.tiff_on_canvas_click)  # For drawing
    
    def upload_tiff_file(self):
        """Upload and display a TIFF file."""
        file_path = filedialog.askopenfilename(
            title="Select TIFF File",
            filetypes=[
                ("GeoTIFF files", "*.tif *.tiff"),
                ("ZIP files", "*.zip"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.tiff_status_label.config(text="Loading TIFF file...")
        self.root.update()
        
        try:
            tif_path = file_path
            
            # If ZIP, extract and find GeoTIFF
            if file_path.lower().endswith('.zip'):
                extract_dir = os.path.join(os.path.dirname(file_path), "tiff_extracted")
                os.makedirs(extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find .tif file
                tif_files = [f for f in os.listdir(extract_dir) if f.lower().endswith(('.tif', '.tiff'))]
                if not tif_files:
                    messagebox.showerror("Error", "No GeoTIFF file found in ZIP archive.")
                    return
                tif_path = os.path.join(extract_dir, tif_files[0])
            
            # Load TIFF using rasterio
            try:
                import rasterio
                from rasterio.plot import show
                
                with rasterio.open(tif_path) as dataset:
                    # Read the data
                    data = dataset.read()
                    
                    # Get bounds for reference
                    bounds = dataset.bounds
                    crs = dataset.crs
                    
                    # Convert to PIL Image for display
                    if len(data.shape) == 3:  # Multi-band
                        # Use first 3 bands as RGB if available
                        if data.shape[0] >= 3:
                            # Normalize to 0-255
                            r = (data[0] / data[0].max() * 255).astype(np.uint8) if data[0].max() > 0 else data[0].astype(np.uint8)
                            g = (data[1] / data[1].max() * 255).astype(np.uint8) if data[1].max() > 0 else data[1].astype(np.uint8)
                            b = (data[2] / data[2].max() * 255).astype(np.uint8) if data[2].max() > 0 else data[2].astype(np.uint8)
                            img_array = np.dstack((r, g, b))
                        else:
                            # Only 1-2 bands, grayscale
                            band = (data[0] / data[0].max() * 255).astype(np.uint8) if data[0].max() > 0 else data[0].astype(np.uint8)
                            img_array = np.dstack((band, band, band))
                    else:  # Single band
                        band = (data / data.max() * 255).astype(np.uint8) if data.max() > 0 else data.astype(np.uint8)
                        img_array = np.dstack((band, band, band))
                    
                    self.tiff_image = PIL.Image.fromarray(img_array.astype(np.uint8))
                    self.tiff_path = tif_path
                    self.tiff_zoom = 1.0
                    self.tiff_pan_x = 0
                    self.tiff_pan_y = 0
                    self.tiff_zoom_cache.clear()  # Clear cache on new load
                    
                    # Initialize drawing tools for TIFF
                    if self.tiff_drawing_tools is None:
                        self.tiff_drawing_tools = DrawingToolsManager(None)
                    else:
                        self.tiff_drawing_tools.clear_all()
                    
                    # Display
                    self.display_tiff()
                    
                    info_text = f"TIFF loaded: {os.path.basename(tif_path)}\n"
                    info_text += f"Size: {self.tiff_image.width}x{self.tiff_image.height}\n"
                    info_text += f"CRS: {crs if crs else 'Unknown'}"
                    
                    messagebox.showinfo("TIFF Loaded", info_text)
                    self.tiff_status_label.config(text=f"Loaded: {os.path.basename(tif_path)}")
                    
            except ImportError:
                messagebox.showerror("Error", "rasterio not installed. Cannot process GeoTIFF.")
            except Exception as e:
                messagebox.showerror("Error", f"Error processing TIFF: {e}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading TIFF file: {e}")
            self.tiff_status_label.config(text="Error loading file")
    
    def display_tiff(self):
        """Display the TIFF image on the canvas with optimized caching."""
        if self.tiff_image is None or self.tiff_canvas is None:
            return
        
        try:
            canvas_width = self.tiff_canvas.winfo_width()
            canvas_height = self.tiff_canvas.winfo_height()
            
            if canvas_width < 100 or canvas_height < 100:
                self.root.after(50, self.display_tiff)
                return
            
            # Use cached image if available
            zoom_key = round(self.tiff_zoom, 2)
            if zoom_key not in self.tiff_zoom_cache:
                display_width = max(1, int(self.tiff_image.width * self.tiff_zoom))
                display_height = max(1, int(self.tiff_image.height * self.tiff_zoom))
                
                if self.tiff_zoom != 1.0:
                    display_image = self.tiff_image.resize((display_width, display_height), PIL.Image.Resampling.FAST)
                else:
                    display_image = self.tiff_image
                
                self.tiff_zoom_cache[zoom_key] = display_image
            else:
                display_image = self.tiff_zoom_cache[zoom_key]
            
            # Clear cache if it gets too large
            if len(self.tiff_zoom_cache) > 10:
                oldest_key = list(self.tiff_zoom_cache.keys())[0]
                del self.tiff_zoom_cache[oldest_key]
            
            # Convert to PhotoImage
            self.tiff_photo_image = PIL.ImageTk.PhotoImage(display_image)
            
            # Update canvas
            self.tiff_canvas.delete("all")
            self.tiff_canvas.create_image(self.tiff_pan_x, self.tiff_pan_y, image=self.tiff_photo_image, anchor=tk.NW)
            
            # Draw annotations if any
            self.draw_tiff_annotations()
            
            self.tiff_status_label.config(text=f"Zoom: {self.tiff_zoom:.2f}x | Pan: ({self.tiff_pan_x}, {self.tiff_pan_y})")
        except Exception as e:
            self.tiff_status_label.config(text=f"Error: {e}")
    
    def draw_tiff_annotations(self):
        """Draw annotation overlays on TIFF canvas."""
        if not self.tiff_drawing_tools or not self.tiff_drawing_tools.drawn_objects:
            return
        
        try:
            for obj_id, obj_data in self.tiff_drawing_tools.drawn_objects.items():
                color = obj_data['color']
                coords = obj_data['coords']
                
                if obj_data['type'] == 'marker':
                    # Draw a circle marker
                    x, y = coords[0]
                    x_canvas = x * self.tiff_zoom + self.tiff_pan_x
                    y_canvas = y * self.tiff_zoom + self.tiff_pan_y
                    r = max(5, int(10 * self.tiff_zoom))
                    self.tiff_canvas.create_oval(x_canvas-r, y_canvas-r, x_canvas+r, y_canvas+r, outline=color, width=2)
                    self.tiff_canvas.create_text(x_canvas+r+5, y_canvas-r, text=obj_data['label'], fill=color, font=("Arial", 8))
                
                elif obj_data['type'] == 'line':
                    # Draw line segments
                    for i in range(len(coords)-1):
                        x1, y1 = coords[i]
                        x2, y2 = coords[i+1]
                        x1_c = x1 * self.tiff_zoom + self.tiff_pan_x
                        y1_c = y1 * self.tiff_zoom + self.tiff_pan_y
                        x2_c = x2 * self.tiff_zoom + self.tiff_pan_x
                        y2_c = y2 * self.tiff_zoom + self.tiff_pan_y
                        self.tiff_canvas.create_line(x1_c, y1_c, x2_c, y2_c, fill=color, width=2)
                
                elif obj_data['type'] == 'polygon':
                    # Draw polygon
                    if len(coords) >= 3:
                        canvas_coords = []
                        for x, y in coords:
                            canvas_coords.append(x * self.tiff_zoom + self.tiff_pan_x)
                            canvas_coords.append(y * self.tiff_zoom + self.tiff_pan_y)
                        # Close the polygon
                        canvas_coords.append(canvas_coords[0])
                        canvas_coords.append(canvas_coords[1])
                        self.tiff_canvas.create_polygon(canvas_coords, outline=color, fill="", width=2)
        except:
            pass
    
    def on_tiff_drawing_mode_changed(self, event=None):
        """Handle TIFF drawing mode change."""
        if not self.tiff_drawing_tools:
            self.tiff_drawing_tools = DrawingToolsManager(None)  # Create dummy object for canvas drawing
        
        mode = self.tiff_drawing_mode_var.get()
        self.tiff_drawing_tools.set_drawing_mode(mode)
        
        if mode in ["Line", "Polygon"]:
            self.tiff_finish_drawing_btn.config(state="normal")
        else:
            self.tiff_finish_drawing_btn.config(state="disabled")
    
    def on_tiff_color_changed(self, event=None):
        """Handle TIFF color change."""
        if self.tiff_drawing_tools:
            color_name = self.tiff_color_var.get()
            self.tiff_drawing_tools.set_color(color_name)
    
    def finish_tiff_drawing(self):
        """Finish drawing a line or polygon on TIFF."""
        try:
            if not self.tiff_drawing_tools:
                self.tiff_status_label.config(text="Error: Drawing tools not initialized")
                return
            
            if self.tiff_drawing_tools.finish_line_or_polygon():
                self.tiff_drawing_mode_var.set("None")
                self.on_tiff_drawing_mode_changed()
                self.display_tiff()
            else:
                self.tiff_status_label.config(text="Need at least 2 points")
        except Exception as e:
            print(f"Error finishing TIFF drawing: {e}")
            self.tiff_status_label.config(text=f"Error: {e}")
    
    def tiff_on_canvas_click(self, event):
        """Handle canvas click for TIFF drawing."""
        if self.tiff_image is None or not self.tiff_drawing_tools:
            return
        
        # Convert canvas coords to image coords
        img_x = (event.x - self.tiff_pan_x) / self.tiff_zoom
        img_y = (event.y - self.tiff_pan_y) / self.tiff_zoom
        
        if img_x < 0 or img_y < 0 or img_x > self.tiff_image.width or img_y > self.tiff_image.height:
            return
        
        mode = self.tiff_drawing_mode_var.get()
        
        if mode == "Marker":
            self.tiff_drawing_tools.add_marker(img_x, img_y, label=f"Mark {len(self.tiff_drawing_tools.drawn_objects)+1}")
            self.display_tiff()
        elif mode in ["Line", "Polygon"]:
            self.tiff_drawing_tools.temp_points.append((img_x, img_y))
            self.display_tiff()
    
    def clear_tiff_drawings(self):
        """Clear all TIFF drawings."""
        if self.tiff_drawing_tools:
            self.tiff_drawing_tools.clear_all()
            self.display_tiff()
    
    def fit_tiff_to_window(self):
        """Fit the TIFF image to the window (optimized)."""
        if self.tiff_image is None or self.tiff_canvas is None:
            return
        
        canvas_width = self.tiff_canvas.winfo_width()
        canvas_height = self.tiff_canvas.winfo_height()
        
        if canvas_width < 100 or canvas_height < 100:
            return
        
        # Calculate zoom to fit
        zoom_x = canvas_width / self.tiff_image.width
        zoom_y = canvas_height / self.tiff_image.height
        self.tiff_zoom = min(zoom_x, zoom_y) * 0.95
        self.tiff_pan_x = 0
        self.tiff_pan_y = 0
        self.display_tiff()
    
    def zoom_tiff_in(self):
        """Zoom in (optimized)."""
        if self.tiff_image is None:
            return
        self.tiff_zoom = min(self.tiff_zoom * 1.15, 10.0)  # Cap at 10x
        self.display_tiff()
    
    def zoom_tiff_out(self):
        """Zoom out (optimized)."""
        if self.tiff_image is None:
            return
        self.tiff_zoom = max(self.tiff_zoom / 1.15, 0.1)
        self.display_tiff()
    
    def reset_tiff_view(self):
        """Reset TIFF view (optimized)."""
        if self.tiff_image is None:
            return
        self.tiff_zoom = 1.0
        self.tiff_pan_x = 0
        self.tiff_pan_y = 0
        self.tiff_zoom_cache.clear()
        self.display_tiff()
    
    def tiff_mousewheel(self, event):
        """Handle mousewheel zoom (optimized)."""
        if self.tiff_image is None:
            return
        
        if event.num == 5 or event.delta == -120:
            self.zoom_tiff_out()
        elif event.num == 4 or event.delta == 120:
            self.zoom_tiff_in()
    
    def tiff_pan_start(self, event):
        """Start panning (optimized)."""
        self.tiff_pan_start_x = event.x
        self.tiff_pan_start_y = event.y
    
    def tiff_pan(self, event):
        """Pan the TIFF image (optimized)."""
        if self.tiff_image is None:
            return
        
        dx = event.x - self.tiff_pan_start_x
        dy = event.y - self.tiff_pan_start_y
        
        self.tiff_pan_x += dx
        self.tiff_pan_y += dy
        
        self.tiff_pan_start_x = event.x
        self.tiff_pan_start_y = event.y
        
        self.display_tiff()
    
    def clear_all_drawings(self):
        """Clear all drawn objects with confirmation."""
        if not self.drawing_tools or len(self.drawing_tools.drawn_objects) == 0:
            messagebox.showinfo("No Drawings", "No drawn objects to clear.")
            return
        
        if messagebox.askyesno("Clear Drawings", 
                              f"Are you sure you want to delete all {len(self.drawing_tools.drawn_objects)} drawings?"):
            self.drawing_tools.clear_all()
            self.map_status_label.config(text="All drawings cleared")

    
    def change_tile_server(self, event=None):
        """Change the map tile server based on selection."""
        if not self.map_widget:
            return
        
        server = self.tile_server_var.get()
        try:
            if server == "Google Satellite":
                self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)
            elif server == "OpenTopoMap":
                self.map_widget.set_tile_server("https://tile.opentopomap.org/{z}/{x}/{y}.png", max_zoom=17)
            else:  # OpenStreetMap
                self.map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png", max_zoom=19)
            
            self.map_status_label.config(text=f"Map Type: {server}")
        except Exception as e:
            self.map_status_label.config(text=f"Error changing map: {e}")
    
    def clear_all_markers(self):
        """Clear all markers from the map with confirmation."""
        if not self.marker_manager:
            return
        
        if len(self.marker_manager.markers) > 0:
            result = messagebox.askyesno("Clear Markers", 
                                        f"Are you sure you want to delete all {len(self.marker_manager.markers)} markers?")
            if result:
                self.marker_manager.clear_all_markers()
                self.update_marker_count()
                self.map_status_label.config(text="All markers cleared")
    
    def setup_tune_tab(self):
        """Initialize the Tune tab for static fine-tuning of CV  parameters."""
        # Main container with left and right panels
        main_container = tk.Frame(self.tab_tune, bg="#1e1e1e")
        main_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure grid weights for proper resizing
        main_container.grid_columnconfigure(0, weight=2)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)
        
        # Left Panel: Image Display
        left_panel = tk.Frame(main_container, bg="#1e1e1e")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Canvas for tuned image display
        tk.Label(left_panel, text="Static Image Tuning Canvas", bg="#1e1e1e", fg="#00ff00", 
                font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 3))
        
        canvas_frame = tk.Frame(left_panel, bg="black", width=600, height=450)
        canvas_frame.pack(fill="both", expand=True)
        canvas_frame.pack_propagate(False)
        
        self.tune_canvas = tk.Canvas(canvas_frame, bg="black", highlightthickness=0)
        self.tune_canvas.pack(fill="both", expand=True)
        self.tune_canvas.bind("<ButtonPress-1>", self.on_tune_canvas_press)
        self.tune_canvas.bind("<B1-Motion>", self.on_tune_canvas_drag)
        self.tune_canvas.bind("<ButtonRelease-1>", self.on_tune_canvas_release)
        self.tune_canvas.bind("<Double-Button-1>", self.on_tune_canvas_double_click)
        self.tune_canvas.bind("<Button-3>", self.on_tune_canvas_right_click)
        self.tune_canvas.bind("<Configure>", self.on_tune_canvas_configure)
        
        # Initial instruction text on canvas
        self.tune_canvas.create_text(300, 200, 
                                     text="Click '🔧 Tune This' on a saved\ndetection to start tuning",
                                     fill="#00ff00", font=("Arial", 14, "bold"),
                                     justify="center", tags="instruction")
        self.tune_canvas.create_text(300, 250,
                                     text="(Go to 'Saved Detections' tab and capture an ROI first)",
                                     fill="#aaaaaa", font=("Arial", 10),
                                     justify="center", tags="instruction")
        
        # Instructions label
        info_label = tk.Label(left_panel, 
                             text="Adjust sliders to see real-time detection results on the static image.",
                             bg="#1e1e1e", fg="#aaaaaa", font=("Arial", 9), wraplength=600, justify="left")
        info_label.pack(anchor="w", pady=(5, 0))
        
        # Right Panel: Controls (Scrollable)
        right_panel = tk.Frame(main_container, bg="#1e1e1e", width=300)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_panel.grid_propagate(False)
        
        # Scrollable area for sliders
        tune_canvas_scroll = tk.Canvas(right_panel, bg="#1e1e1e", highlightthickness=0)
        tune_scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=tune_canvas_scroll.yview)
        tune_scroll_frame = tk.Frame(tune_canvas_scroll, bg="#1e1e1e")
        
        tune_scroll_frame.bind("<Configure>", 
                              lambda e: tune_canvas_scroll.configure(scrollregion=tune_canvas_scroll.bbox("all")))
        
        tune_canvas_scroll.create_window((0, 0), window=tune_scroll_frame, anchor="nw")
        tune_canvas_scroll.configure(yscrollcommand=tune_scrollbar.set)
        
        tune_canvas_scroll.pack(side="left", fill="both", expand=True)
        tune_scrollbar.pack(side="right", fill="y")
        
        # === MASTER SENSITIVITY SCALAR ===
        master_frame = ttk.LabelFrame(tune_scroll_frame, text="Master Sensitivity", 
                                     style="Section.TLabelframe", padding=10)
        master_frame.pack(fill="x", pady=5)
        
        tk.Label(master_frame, text="Adjust all thresholds proportionally\n(Higher = More Sensitive)", 
                bg="#1e1e1e", fg="#aaaaaa", font=("Arial", 8)).pack(anchor="w", pady=(0, 5))
        
        sensitivity_frame = tk.Frame(master_frame, bg="#1e1e1e")
        sensitivity_frame.pack(fill="x")
        
        tk.Label(sensitivity_frame, text="Master Sensitivity:", bg="#1e1e1e", fg="#aaaaaa", 
                font=("Arial", 9)).pack(anchor="w")
        
        sens_slider = tk.Scale(sensitivity_frame, from_=0.5, to=2.0, resolution=0.1, orient="horizontal",
                              variable=self.master_sensitivity, bg="#1e1e1e", fg="white",
                              troughcolor="#444", highlightthickness=0, bd=0,
                              command=lambda v: self.on_tune_slider_changed())
        sens_slider.pack(fill="x")
        
        sens_value_label = tk.Label(sensitivity_frame, textvariable=self.master_sensitivity, 
                                   bg="#1e1e1e", fg="#00ff00", font=("Arial", 9, "bold"))
        sens_value_label.pack(anchor="w")
        
        # === TUNE SLIDERS (Combined Mode) ===
        # CLAHE Settings
        clahe_frame = ttk.LabelFrame(tune_scroll_frame, text="CLAHE Enhancement", 
                                    style="Section.TLabelframe", padding=5)
        clahe_frame.pack(fill="x", pady=5)
        
        self.tune_slider_clahe_clip = self.create_tune_slider(clahe_frame, "Clip Limit", 1, 10, 3)
        self.tune_slider_clahe_grid = self.create_tune_slider(clahe_frame, "Grid Size", 1, 16, 8)
        
        # Canny Edge Detection
        canny_frame = ttk.LabelFrame(tune_scroll_frame, text="Edge Detection (Canny)", 
                                    style="Section.TLabelframe", padding=5)
        canny_frame.pack(fill="x", pady=5)
        
        self.tune_slider_canny_min = self.create_tune_slider(canny_frame, "Min Gradient", 0, 255, 20)
        self.tune_slider_canny_max = self.create_tune_slider(canny_frame, "Max Gradient", 0, 255, 80)
        self.tune_slider_gauss = self.create_tune_slider(canny_frame, "Gaussian Blur", 1, 15, 3)
        
        # Blob Detection
        blob_frame = ttk.LabelFrame(tune_scroll_frame, text="Blob Detection", 
                                   style="Section.TLabelframe", padding=5)
        blob_frame.pack(fill="x", pady=5)
        
        tk.Checkbutton(blob_frame, text="Use Otsu Thresholding", variable=self.tune_use_otsu,
                      bg="#1e1e1e", fg="#00ff00", selectcolor="#444",
                      activebackground="#1e1e1e", activeforeground="#00ff00",
                      command=self.apply_tune_detection).pack(anchor="w", pady=2)
        
        self.tune_slider_blob_min_thresh = self.create_tune_slider(blob_frame, "Min Threshold", 10, 255, 120)
        self.tune_slider_blob_max_thresh = self.create_tune_slider(blob_frame, "Max Threshold", 10, 255, 255)
        self.tune_slider_blob_min_area = self.create_tune_slider(blob_frame, "Min Area", 10, 500, 25)
        
        # Combined Mode Settings
        combined_frame = ttk.LabelFrame(tune_scroll_frame, text="Edge Filtering", 
                                       style="Section.TLabelframe", padding=5)
        combined_frame.pack(fill="x", pady=5)
        
        tk.Checkbutton(combined_frame, text="Enable Edge Filter", variable=self.tune_enable_edge_filter,
                      bg="#1e1e1e", fg="#00ff00", selectcolor="#444",
                      activebackground="#1e1e1e", activeforeground="#00ff00",
                      command=self.apply_tune_detection).pack(anchor="w", pady=2)
        
        self.tune_slider_edge_density = self.create_tune_slider(combined_frame, "Edge Density Threshold", 1, 100, 3)

        # Calibration controls
        calibration_frame = ttk.LabelFrame(tune_scroll_frame, text="Calibration",
                          style="Section.TLabelframe", padding=5)
        calibration_frame.pack(fill="x", pady=5)

        tk.Label(calibration_frame, text="Target Count", bg="#1e1e1e", fg="#aaaaaa",
             font=("Arial", 8)).pack(anchor="w")
        tk.Entry(calibration_frame, textvariable=self.tune_target_count, bg="#333", fg="white",
             insertbackground="white", relief="flat").pack(fill="x", pady=(0, 5))

        tk.Label(calibration_frame, text="Ground Truth Type", bg="#1e1e1e", fg="#aaaaaa",
             font=("Arial", 8)).pack(anchor="w")
        ttk.Combobox(calibration_frame, values=["Box", "Polygon", "Point"], state="readonly",
                textvariable=self.tune_gt_mode).pack(fill="x", pady=(0, 5))

        tk.Checkbutton(calibration_frame, text="Draw Targets", variable=self.tune_draw_targets,
                  bg="#1e1e1e", fg="#ffff00", selectcolor="#444",
                  activebackground="#1e1e1e", activeforeground="#ffff00",
                  command=self.on_tune_draw_toggle_changed).pack(anchor="w", pady=(0, 5))

        tk.Button(calibration_frame, text="🎯 Auto-Calibrate", command=self.start_auto_calibration,
             bg="#7a42f4", fg="white", font=("Arial", 9, "bold"),
             relief="raised", bd=2).pack(fill="x", pady=(2, 5))

        tk.Label(calibration_frame, textvariable=self.tune_calibration_status, bg="#1e1e1e",
             fg="#00ff00", font=("Arial", 8), justify="left",
             wraplength=250).pack(anchor="w")
        
        # === ACTION BUTTONS ===
        btn_frame = tk.Frame(tune_scroll_frame, bg="#1e1e1e")
        btn_frame.pack(fill="x", pady=10, padx=5)
        
        tk.Button(btn_frame, text="💾 Save as Preset", command=self.save_tune_preset,
                 bg="#5cb85c", fg="white", font=("Arial", 9, "bold"), 
                 height=2, relief="raised", bd=2).pack(fill="x", pady=2)
        
        tk.Button(btn_frame, text="▶ Apply to Live Feed", command=self.apply_tune_to_live,
                 bg="#0275d8", fg="white", font=("Arial", 9, "bold"),
                 height=2, relief="raised", bd=2).pack(fill="x", pady=2)
        
        tk.Button(btn_frame, text="🔄 Reset to Defaults", command=self.reset_tune_sliders,
                 bg="#f0ad4e", fg="white", font=("Arial", 9, "bold"),
                 height=2, relief="raised", bd=2).pack(fill="x", pady=2)
    
    def create_tune_slider(self, parent, label_text, min_val, max_val, default):
        """Create a slider for the Tune tab that triggers detection on change."""
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill="x", pady=2)
        lbl = tk.Label(frame, text=label_text, bg="#1e1e1e", fg="#aaaaaa", font=("Arial", 8))
        lbl.pack(anchor="w")
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient="horizontal", bg="#1e1e1e", fg="white",
                         troughcolor="#444", highlightthickness=0, bd=0,
                         command=lambda v: self.on_tune_slider_changed())
        slider.set(default)
        slider.pack(fill="x")
        return slider

    def on_tune_slider_changed(self):
        if self.tune_skip_slider_callback:
            return
        self.apply_tune_detection()

    def on_tune_draw_toggle_changed(self):
        self.tune_gt_temp_box = None
        self.tune_gt_drag_start = None
        self.draw_tune_overlays()

    def on_tune_canvas_configure(self, event):
        if self.tune_last_render_rgb is not None:
            self.display_tune_image(self.tune_last_render_rgb)

    def _copy_tune_gt_annotations(self, annotations=None):
        src = annotations if annotations is not None else self.tune_gt_annotations
        return {
            'boxes': [tuple(b) for b in src.get('boxes', [])],
            'polygons': [[tuple(pt) for pt in poly] for poly in src.get('polygons', [])],
            'points': [tuple(pt) for pt in src.get('points', [])]
        }

    def _cache_current_tune_gt(self):
        if self.tune_current_image_key is None:
            return
        self.tune_gt_cache[self.tune_current_image_key] = {
            'annotations': self._copy_tune_gt_annotations(),
            'target_count': max(1, int(self.tune_target_count.get() if self.tune_target_count.get() else 1))
        }

    def _load_tune_gt_from_cache(self, image_key):
        cached = self.tune_gt_cache.get(image_key)
        if cached:
            self.tune_gt_annotations = self._copy_tune_gt_annotations(cached.get('annotations', {}))
            self.tune_target_count.set(max(1, int(cached.get('target_count', 1))))
            self.tune_gt_polygon_working = []
            self.tune_gt_temp_box = None
            self.tune_gt_drag_start = None
            return
        self.tune_gt_annotations = {'boxes': [], 'polygons': [], 'points': []}
        self.tune_gt_polygon_working = []
        self.tune_gt_temp_box = None
        self.tune_gt_drag_start = None

    def _source_to_canvas(self, sx, sy):
        t = self.tune_display_transform
        scale = t.get('scale', 1.0)
        return (sx * scale + t.get('offset_x', 0), sy * scale + t.get('offset_y', 0))

    def _canvas_to_source(self, cx, cy):
        t = self.tune_display_transform
        scale = t.get('scale', 1.0)
        if scale <= 0:
            return None
        sx = (cx - t.get('offset_x', 0)) / scale
        sy = (cy - t.get('offset_y', 0)) / scale
        img_w = t.get('img_w', 0)
        img_h = t.get('img_h', 0)
        if img_w <= 0 or img_h <= 0:
            return None
        sx = int(np.clip(sx, 0, img_w - 1))
        sy = int(np.clip(sy, 0, img_h - 1))
        return (sx, sy)

    def _event_inside_display_image(self, event):
        t = self.tune_display_transform
        x0, y0 = t.get('offset_x', 0), t.get('offset_y', 0)
        x1, y1 = x0 + t.get('draw_w', 0), y0 + t.get('draw_h', 0)
        return x0 <= event.x <= x1 and y0 <= event.y <= y1

    def on_tune_canvas_press(self, event):
        if not self.tune_draw_targets.get() or self.tune_image is None or self.tune_is_calibrating:
            return
        if not self._event_inside_display_image(event):
            return
        if self.tune_gt_mode.get() != "Box":
            return
        src = self._canvas_to_source(event.x, event.y)
        if src is None:
            return
        self.tune_gt_drag_start = src
        self.tune_gt_temp_box = (src[0], src[1], src[0], src[1])
        self.draw_tune_overlays()

    def on_tune_canvas_drag(self, event):
        if not self.tune_draw_targets.get() or self.tune_gt_mode.get() != "Box":
            return
        if self.tune_gt_drag_start is None:
            return
        src = self._canvas_to_source(event.x, event.y)
        if src is None:
            return
        x0, y0 = self.tune_gt_drag_start
        self.tune_gt_temp_box = (x0, y0, src[0], src[1])
        self.draw_tune_overlays()

    def on_tune_canvas_release(self, event):
        if not self.tune_draw_targets.get() or self.tune_image is None or self.tune_is_calibrating:
            return
        if not self._event_inside_display_image(event):
            return

        mode = self.tune_gt_mode.get()
        src = self._canvas_to_source(event.x, event.y)
        if src is None:
            return

        if mode == "Box" and self.tune_gt_drag_start is not None:
            x0, y0 = self.tune_gt_drag_start
            x1, y1 = src
            bx0, bx1 = sorted([x0, x1])
            by0, by1 = sorted([y0, y1])
            if (bx1 - bx0) >= 5 and (by1 - by0) >= 5:
                self.tune_gt_annotations['boxes'].append((bx0, by0, bx1 - bx0, by1 - by0))
                self._cache_current_tune_gt()
            self.tune_gt_drag_start = None
            self.tune_gt_temp_box = None
        elif mode == "Point":
            self.tune_gt_annotations['points'].append((src[0], src[1]))
            self._cache_current_tune_gt()
        elif mode == "Polygon":
            self.tune_gt_polygon_working.append((src[0], src[1]))

        self.draw_tune_overlays()

    def on_tune_canvas_double_click(self, event):
        if not self.tune_draw_targets.get() or self.tune_is_calibrating:
            return
        if self.tune_gt_mode.get() != "Polygon":
            return
        if len(self.tune_gt_polygon_working) >= 3:
            self.tune_gt_annotations['polygons'].append(list(self.tune_gt_polygon_working))
            self._cache_current_tune_gt()
        self.tune_gt_polygon_working = []
        self.draw_tune_overlays()

    def on_tune_canvas_right_click(self, event):
        if self.tune_is_calibrating:
            return
        mode = self.tune_gt_mode.get()
        if mode == "Polygon" and self.tune_gt_polygon_working:
            self.tune_gt_polygon_working.pop()
            self.draw_tune_overlays()
            return
        src = self._canvas_to_source(event.x, event.y)
        if src is None:
            return
        sx, sy = src
        if mode == "Point" and self.tune_gt_annotations['points']:
            distances = [((px - sx) ** 2 + (py - sy) ** 2, idx) for idx, (px, py) in enumerate(self.tune_gt_annotations['points'])]
            _, idx = min(distances)
            self.tune_gt_annotations['points'].pop(idx)
            self._cache_current_tune_gt()
        elif mode == "Box" and self.tune_gt_annotations['boxes']:
            box_dist = []
            for idx, (x, y, w, h) in enumerate(self.tune_gt_annotations['boxes']):
                cx, cy = x + w / 2.0, y + h / 2.0
                box_dist.append((((cx - sx) ** 2 + (cy - sy) ** 2), idx))
            _, idx = min(box_dist)
            self.tune_gt_annotations['boxes'].pop(idx)
            self._cache_current_tune_gt()
        elif mode == "Polygon" and self.tune_gt_annotations['polygons']:
            poly_dist = []
            for idx, poly in enumerate(self.tune_gt_annotations['polygons']):
                pts = np.array(poly, dtype=np.float32)
                cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
                poly_dist.append((((cx - sx) ** 2 + (cy - sy) ** 2), idx))
            _, idx = min(poly_dist)
            self.tune_gt_annotations['polygons'].pop(idx)
            self._cache_current_tune_gt()
        self.draw_tune_overlays()

    def draw_tune_overlays(self):
        if self.tune_canvas is None:
            return
        self.tune_canvas.delete("gt_overlay")
        t = self.tune_display_transform
        if t.get('draw_w', 0) <= 0 or t.get('draw_h', 0) <= 0:
            return

        # Draw persisted yellow boxes
        for (x, y, w, h) in self.tune_gt_annotations.get('boxes', []):
            x0, y0 = self._source_to_canvas(x, y)
            x1, y1 = self._source_to_canvas(x + w, y + h)
            self.tune_canvas.create_rectangle(x0, y0, x1, y1, outline="#ffff00", width=2, tags="gt_overlay")

        # Draw persisted yellow polygons
        for polygon in self.tune_gt_annotations.get('polygons', []):
            if len(polygon) < 2:
                continue
            canvas_pts = []
            for px, py in polygon:
                cx, cy = self._source_to_canvas(px, py)
                canvas_pts.extend([cx, cy])
            self.tune_canvas.create_polygon(*canvas_pts, outline="#ffff00", fill="", width=2, tags="gt_overlay")

        # Draw persisted yellow points
        for (px, py) in self.tune_gt_annotations.get('points', []):
            cx, cy = self._source_to_canvas(px, py)
            self.tune_canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, outline="#ffff00", fill="#ffff00", tags="gt_overlay")

        # Draw in-progress polygon
        if len(self.tune_gt_polygon_working) >= 1:
            canvas_pts = []
            for px, py in self.tune_gt_polygon_working:
                cx, cy = self._source_to_canvas(px, py)
                canvas_pts.extend([cx, cy])
            if len(canvas_pts) >= 4:
                self.tune_canvas.create_line(*canvas_pts, fill="#ffff00", width=2, tags="gt_overlay")

        # Draw temporary drag box
        if self.tune_gt_temp_box is not None:
            x0, y0, x1, y1 = self.tune_gt_temp_box
            cx0, cy0 = self._source_to_canvas(x0, y0)
            cx1, cy1 = self._source_to_canvas(x1, y1)
            self.tune_canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="#ffff00", width=2, dash=(4, 2), tags="gt_overlay")
    
    def load_image_for_tuning(self, roi_data):
        """Load a static image into the Tune tab."""
        try:
            self._cache_current_tune_gt()
            self.tune_image = roi_data['full_frame'].copy()
            self.tune_roi_bbox = roi_data['roi_bbox']  # Store ROI bbox (x, y, w, h)
            self.tune_current_image_key = id(roi_data)
            self._load_tune_gt_from_cache(self.tune_current_image_key)
            self.tune_calibration_status.set("Idle")
            self.notebook.select(self.tab_tune)  # Switch to Tune tab
            # Force canvas update to get proper dimensions
            self.tune_canvas.update_idletasks()
            self.apply_tune_detection()  # Run detection immediately
        except Exception as e:
            print(f"Error loading image for tuning: {e}")
            messagebox.showerror("Tune Error", f"Failed to load image: {e}")

    def _get_current_tune_settings(self):
        return {
            'master_sensitivity': float(self.master_sensitivity.get()),
            'clahe_clip': float(self.tune_slider_clahe_clip.get()),
            'clahe_grid': int(self.tune_slider_clahe_grid.get()),
            'canny_min': int(self.tune_slider_canny_min.get()),
            'canny_max': int(self.tune_slider_canny_max.get()),
            'gauss': int(self.tune_slider_gauss.get()),
            'blob_min_thresh': int(self.tune_slider_blob_min_thresh.get()),
            'blob_max_thresh': int(self.tune_slider_blob_max_thresh.get()),
            'blob_min_area': int(self.tune_slider_blob_min_area.get()),
            'edge_density': int(self.tune_slider_edge_density.get()),
            'use_otsu': bool(self.tune_use_otsu.get()),
            'enable_edge_filter': bool(self.tune_enable_edge_filter.get())
        }

    def _run_tune_detection_with_settings(self, frame_rgb, roi_bbox, settings, include_render=True):
        frame = frame_rgb.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        height, width = frame.shape[:2]
        roi_x, roi_y, roi_w, roi_h = roi_bbox

        sensitivity = max(0.1, float(settings['master_sensitivity']))
        canny_min = max(1, int(settings['canny_min'] / sensitivity))
        canny_max = max(1, int(settings['canny_max'] / sensitivity))
        blob_min_thresh = max(1, int(settings['blob_min_thresh'] / sensitivity))
        blob_max_thresh = int(settings['blob_max_thresh'])
        blob_min_area = max(1, int(settings['blob_min_area'] / sensitivity))

        lab = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = float(settings['clahe_clip'])
        grid = int(settings['clahe_grid'])
        grid = max(1, grid)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        filter_edges = bool(settings['enable_edge_filter'])
        edges = None
        if filter_edges:
            k_size = int(settings['gauss'])
            if k_size % 2 == 0:
                k_size += 1
            k_size = max(1, k_size)
            blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)
            edges = cv2.Canny(blurred, canny_min, canny_max)

        params = cv2.SimpleBlobDetector_Params()
        if bool(settings['use_otsu']):
            otsu_val, binary_map = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            blob_min_thresh = otsu_val
            blob_max_thresh = 255
        else:
            _, binary_map = cv2.threshold(gray, blob_min_thresh, 255, cv2.THRESH_BINARY)

        if blob_min_thresh >= blob_max_thresh:
            blob_max_thresh = blob_min_thresh + 1

        params.minThreshold = blob_min_thresh
        params.maxThreshold = blob_max_thresh
        params.thresholdStep = 10
        params.minRepeatability = 1
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = blob_min_area
        params.maxArea = 50000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        detected_rects = []
        min_edge_density = float(settings['edge_density']) / 100.0

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size * 0.8)
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(width, x + r)
            y2 = min(height, y + r)

            roi_bin = binary_map[y1:y2, x1:x2]
            found_rect = None
            if roi_bin.size > 0:
                contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(c)
                    found_rect = (x1 + bx, y1 + by, bw, bh)

            if not found_rect:
                r_std = int(kp.size / 2)
                found_rect = (max(0, x - r_std), max(0, y - r_std), r_std * 2, r_std * 2)

            is_valid = True
            if filter_edges and edges is not None:
                fx, fy, fw, fh = found_rect
                roi_edges = edges[fy:fy + fh, fx:fx + fw]
                if roi_edges.size > 0:
                    density = np.count_nonzero(roi_edges) / roi_edges.size
                    if density < min_edge_density:
                        is_valid = False
                else:
                    is_valid = False

            if is_valid:
                dx, dy, dw, dh = found_rect
                det_center_x = dx + dw // 2
                det_center_y = dy + dh // 2
                if (roi_x <= det_center_x <= roi_x + roi_w and roi_y <= det_center_y <= roi_y + roi_h):
                    detected_rects.append(found_rect)

        result = {
            'detected_rects': detected_rects,
            'detected_count': len(detected_rects),
            'gray': gray,
            'binary_map': binary_map,
            'edges': edges
        }

        if include_render:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if edges is not None:
                edge_mask = edges > 0
                frame_bgr[edge_mask] = [0, 255, 255]
            cv2.rectangle(frame_bgr, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 3)
            for (x, y, w, h) in detected_rects:
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"ROI Detections: {len(detected_rects)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Sensitivity: {sensitivity:.1f}x", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            result['frame_rgb'] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        return result
    
    def apply_tune_detection(self):
        """Apply Combined detection to the static tuning image with current slider settings."""
        if self.tune_image is None or not hasattr(self, 'tune_roi_bbox'):
            return
        result = self._run_tune_detection_with_settings(
            self.tune_image,
            self.tune_roi_bbox,
            self._get_current_tune_settings(),
            include_render=True
        )
        self.tune_last_detected_rects = result['detected_rects']
        frame_rgb = result.get('frame_rgb')
        if frame_rgb is not None:
            self.display_tune_image(frame_rgb)
    
    def display_tune_image(self, img_array):
        """Display an image on the tune canvas."""
        try:
            self.tune_last_render_rgb = img_array.copy()
            img_pil = PIL.Image.fromarray(img_array)
            
            # Force update to get actual dimensions
            self.tune_canvas.update_idletasks()
            canvas_width = self.tune_canvas.winfo_width()
            canvas_height = self.tune_canvas.winfo_height()
            
            # Use reasonable defaults if canvas not yet rendered
            if canvas_width <= 1:
                canvas_width = 600
            if canvas_height <= 1:
                canvas_height = 450
            
            max_w = max(1, canvas_width - 10)
            max_h = max(1, canvas_height - 10)
            src_w, src_h = img_pil.size
            scale = min(max_w / src_w, max_h / src_h)
            draw_w = max(1, int(src_w * scale))
            draw_h = max(1, int(src_h * scale))
            img_pil = img_pil.resize((draw_w, draw_h), PIL.Image.Resampling.LANCZOS)

            offset_x = (canvas_width - draw_w) // 2
            offset_y = (canvas_height - draw_h) // 2
            self.tune_display_transform = {
                'scale': scale,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'draw_w': draw_w,
                'draw_h': draw_h,
                'img_w': src_w,
                'img_h': src_h
            }
            
            self.tune_photo = PIL.ImageTk.PhotoImage(img_pil)
            self.tune_canvas.delete("all")  # This removes instruction text too
            self.tune_canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.tune_photo)
            self.draw_tune_overlays()
        except Exception as e:
            print(f"Error displaying tune image: {e}")
            # Restore instruction text on error
            self.tune_canvas.create_text(300, 225, 
                                         text=f"Error loading image: {str(e)}",
                                         fill="#ff0000", font=("Arial", 12, "bold"),
                                         justify="center")
    
    def save_tune_preset(self):
        """Save current tune settings as a named preset to JSON file."""
        preset_name = simpledialog.askstring("Save Preset", "Enter preset name:")
        if not preset_name:
            return
        
        # Gather all slider values
        preset_data = {
            'clahe_clip': self.tune_slider_clahe_clip.get(),
            'clahe_grid': self.tune_slider_clahe_grid.get(),
            'canny_min': self.tune_slider_canny_min.get(),
            'canny_max': self.tune_slider_canny_max.get(),
            'gauss_blur': self.tune_slider_gauss.get(),
            'blob_min_thresh': self.tune_slider_blob_min_thresh.get(),
            'blob_max_thresh': self.tune_slider_blob_max_thresh.get(),
            'blob_min_area': self.tune_slider_blob_min_area.get(),
            'edge_density': self.tune_slider_edge_density.get(),
            'use_otsu': self.tune_use_otsu.get(),
            'enable_edge_filter': self.tune_enable_edge_filter.get(),
            'master_sensitivity': self.master_sensitivity.get()
        }
        
        # Load existing presets or create new file
        import json
        preset_path = "drone_cv_presets.json"
        
        try:
            if os.path.exists(preset_path):
                with open(preset_path, 'r') as f:
                    presets = json.load(f)
            else:
                presets = {}
            
            presets[preset_name] = preset_data
            
            with open(preset_path, 'w') as f:
                json.dump(presets, f, indent=2)
            
            messagebox.showinfo("Preset Saved", f"Preset '{preset_name}' saved successfully!")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save preset: {e}")
    
    def apply_tune_to_live(self):
        """Apply current tune settings to the live feed sliders."""
        # Copy tune slider values to main sliders
        self.slider_clahe_clip.set(self.tune_slider_clahe_clip.get())
        self.slider_clahe_grid.set(self.tune_slider_clahe_grid.get())
        self.slider_canny_min.set(self.tune_slider_canny_min.get())
        self.slider_canny_max.set(self.tune_slider_canny_max.get())
        self.slider_gauss.set(self.tune_slider_gauss.get())
        self.slider_blob_min_thresh.set(self.tune_slider_blob_min_thresh.get())
        self.slider_blob_max_thresh.set(self.tune_slider_blob_max_thresh.get())
        self.slider_blob_min_area.set(self.tune_slider_blob_min_area.get())
        self.slider_edge_density.set(self.tune_slider_edge_density.get())
        self.use_otsu.set(self.tune_use_otsu.get())
        self.enable_edge_filter.set(self.tune_enable_edge_filter.get())
        
        # Switch to Combined mode
        self.cv_method.set(5)
        
        messagebox.showinfo("Applied", "Tune settings applied to live video feed! Mode set to 'Combined'.")
    
    def reset_tune_sliders(self):
        """Reset all tune sliders to default values."""
        self.tune_skip_slider_callback = True
        self.tune_slider_clahe_clip.set(3)
        self.tune_slider_clahe_grid.set(8)
        self.tune_slider_canny_min.set(20)
        self.tune_slider_canny_max.set(80)
        self.tune_slider_gauss.set(3)
        self.tune_slider_blob_min_thresh.set(120)
        self.tune_slider_blob_max_thresh.set(255)
        self.tune_slider_blob_min_area.set(25)
        self.tune_slider_edge_density.set(3)
        self.tune_skip_slider_callback = False
        self.tune_use_otsu.set(True)
        self.tune_enable_edge_filter.set(True)
        self.master_sensitivity.set(1.0)
        
        if self.tune_image is not None:
            self.apply_tune_detection()

    def _set_tune_settings(self, settings):
        self.tune_skip_slider_callback = True
        self.master_sensitivity.set(float(settings['master_sensitivity']))
        self.tune_slider_canny_min.set(int(settings['canny_min']))
        self.tune_slider_canny_max.set(int(settings['canny_max']))
        self.tune_slider_edge_density.set(int(settings['edge_density']))
        self.tune_slider_blob_min_thresh.set(int(settings['blob_min_thresh']))
        self.tune_slider_blob_min_area.set(int(settings['blob_min_area']))
        self.tune_skip_slider_callback = False

    def _rect_iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        union_area = (aw * ah) + (bw * bh) - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _box_match_score(self, gt_boxes, det_boxes, iou_threshold=0.3):
        if len(gt_boxes) == 0:
            return None
        if len(det_boxes) == 0:
            return 0.0
        candidates = []
        for gi, g in enumerate(gt_boxes):
            for di, d in enumerate(det_boxes):
                iou = self._rect_iou(g, d)
                if iou >= iou_threshold:
                    candidates.append((iou, gi, di))
        candidates.sort(reverse=True, key=lambda x: x[0])
        used_g = set()
        used_d = set()
        score_sum = 0.0
        for iou, gi, di in candidates:
            if gi in used_g or di in used_d:
                continue
            used_g.add(gi)
            used_d.add(di)
            score_sum += iou
        return float(np.clip(score_sum / max(1, len(gt_boxes)), 0.0, 1.0))

    def _polygon_mask_iou_score(self, gt_polygons, det_boxes, shape):
        if len(gt_polygons) == 0:
            return None
        h, w = shape
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        det_mask = np.zeros((h, w), dtype=np.uint8)
        for poly in gt_polygons:
            if len(poly) < 3:
                continue
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(gt_mask, [pts], 1)
        for x, y, bw, bh in det_boxes:
            cv2.rectangle(det_mask, (x, y), (x + bw, y + bh), 1, -1)
        inter = np.logical_and(gt_mask > 0, det_mask > 0).sum()
        union = np.logical_or(gt_mask > 0, det_mask > 0).sum()
        if union <= 0:
            return 0.0
        return float(inter / union)

    def _point_match_score(self, gt_points, det_boxes, radius_px=20):
        if len(gt_points) == 0:
            return None
        if len(det_boxes) == 0:
            return 0.0
        det_centers = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in det_boxes]
        used = set()
        total = 0.0
        for px, py in gt_points:
            distances = [((cx - px) ** 2 + (cy - py) ** 2, idx) for idx, (cx, cy) in enumerate(det_centers) if idx not in used]
            if not distances:
                continue
            d2, idx = min(distances)
            used.add(idx)
            d = np.sqrt(d2)
            total += max(0.0, 1.0 - (d / max(1.0, radius_px)))
        return float(np.clip(total / max(1, len(gt_points)), 0.0, 1.0))

    def _calibration_objective(self, det_boxes, target_count, gt_annotations, frame_shape):
        count_err = abs(len(det_boxes) - target_count) / max(1, target_count)

        box_score = self._box_match_score(gt_annotations.get('boxes', []), det_boxes, iou_threshold=0.30)
        poly_score = self._polygon_mask_iou_score(gt_annotations.get('polygons', []), det_boxes, frame_shape)
        point_score = self._point_match_score(gt_annotations.get('points', []), det_boxes, radius_px=20)

        active_scores = [s for s in [box_score, poly_score, point_score] if s is not None]
        geometry_score = float(np.mean(active_scores)) if active_scores else 0.0
        objective = count_err + (1.0 - geometry_score)
        return objective, geometry_score, count_err

    def _estimate_initial_thresholds_from_gt(self, gray_frame, gt_annotations):
        intensities = []
        areas = []

        for x, y, w, h in gt_annotations.get('boxes', []):
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(gray_frame.shape[1], int(x + w)), min(gray_frame.shape[0], int(y + h))
            if x1 > x0 and y1 > y0:
                crop = gray_frame[y0:y1, x0:x1]
                if crop.size > 0:
                    intensities.append(crop.flatten())
                    areas.append((x1 - x0) * (y1 - y0))

        for poly in gt_annotations.get('polygons', []):
            if len(poly) < 3:
                continue
            mask = np.zeros(gray_frame.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
            vals = gray_frame[mask > 0]
            if vals.size > 0:
                intensities.append(vals)
            areas.append(float(cv2.contourArea(np.array(poly, dtype=np.float32))))

        for px, py in gt_annotations.get('points', []):
            x0, y0 = max(0, int(px) - 5), max(0, int(py) - 5)
            x1, y1 = min(gray_frame.shape[1], int(px) + 6), min(gray_frame.shape[0], int(py) + 6)
            crop = gray_frame[y0:y1, x0:x1]
            if crop.size > 0:
                intensities.append(crop.flatten())
                areas.append(20 * 20)

        if not intensities:
            return None, None

        all_vals = np.concatenate(intensities)
        min_int = float(np.min(all_vals))
        max_int = float(np.max(all_vals))
        init_thresh = int(np.clip((min_int + max_int) * 0.5, 10, 255))

        avg_area = float(np.mean(areas)) if areas else 100.0
        init_min_area = int(np.clip(avg_area * 0.15, 10, 500))
        return init_thresh, init_min_area

    def _build_calibration_search_space(self, base_settings):
        base_master = float(base_settings['master_sensitivity'])
        base_edge = int(base_settings['edge_density'])
        base_cmin = int(base_settings['canny_min'])
        base_cmax = int(base_settings['canny_max'])

        coarse_master = np.linspace(max(0.5, base_master * 0.7), min(2.0, base_master * 1.4), 5)
        coarse_edge = np.linspace(max(1, base_edge - 30), min(100, base_edge + 30), 5)
        coarse_cmin = np.linspace(max(0, base_cmin - 45), min(255, base_cmin + 45), 5)
        coarse_cmax = np.linspace(max(1, base_cmax - 60), min(255, base_cmax + 60), 5)

        candidates = []
        for m in coarse_master:
            for e in coarse_edge:
                for cmin in coarse_cmin:
                    for cmax in coarse_cmax:
                        if cmax <= cmin + 5:
                            continue
                        cand = dict(base_settings)
                        cand['master_sensitivity'] = float(np.clip(m, 0.5, 2.0))
                        cand['edge_density'] = int(np.clip(round(e), 1, 100))
                        cand['canny_min'] = int(np.clip(round(cmin), 0, 255))
                        cand['canny_max'] = int(np.clip(round(cmax), 1, 255))
                        candidates.append(cand)
        return candidates

    def _auto_calibration_worker(self, snapshot):
        try:
            frame = snapshot['frame']
            roi_bbox = snapshot['roi_bbox']
            gt_annotations = snapshot['annotations']
            target_count = snapshot['target_count']
            base_settings = snapshot['base_settings']
            img_shape = frame.shape[:2]

            candidates = self._build_calibration_search_space(base_settings)
            best_score = float('inf')
            best_settings = dict(base_settings)
            total = len(candidates)

            for idx, settings in enumerate(candidates):
                result = self._run_tune_detection_with_settings(frame, roi_bbox, settings, include_render=False)
                objective, geom_score, count_err = self._calibration_objective(
                    result['detected_rects'],
                    target_count,
                    gt_annotations,
                    img_shape
                )

                if objective < best_score:
                    best_score = objective
                    best_settings = dict(settings)
                    self.tune_calibration_queue.put({
                        'type': 'best',
                        'settings': dict(best_settings),
                        'score': float(best_score),
                        'geometry': float(geom_score),
                        'count_error': float(count_err),
                        'progress': idx + 1,
                        'total': total
                    })

                if (idx + 1) % 20 == 0:
                    self.tune_calibration_queue.put({'type': 'progress', 'progress': idx + 1, 'total': total})

            self.tune_calibration_queue.put({'type': 'done', 'settings': best_settings, 'score': float(best_score)})
        except Exception as exc:
            self.tune_calibration_queue.put({'type': 'error', 'error': str(exc)})

    def _poll_auto_calibration(self):
        if not self.tune_is_calibrating or self.tune_calibration_queue is None:
            return
        try:
            while True:
                msg = self.tune_calibration_queue.get_nowait()
                msg_type = msg.get('type')
                if msg_type == 'progress':
                    self.tune_calibration_status.set(f"Calibrating... {msg.get('progress', 0)}/{msg.get('total', 0)}")
                elif msg_type == 'best':
                    self._set_tune_settings(msg['settings'])
                    self.apply_tune_detection()
                    self.tune_calibration_status.set(
                        f"Best so far @ {msg.get('progress', 0)}/{msg.get('total', 0)} | "
                        f"Score {msg.get('score', 0.0):.3f}"
                    )
                elif msg_type == 'done':
                    self._set_tune_settings(msg['settings'])
                    self.apply_tune_detection()
                    self.tune_is_calibrating = False
                    self.tune_calibration_status.set(f"Calibration complete. Final score: {msg.get('score', 0.0):.3f}")
                    return
                elif msg_type == 'error':
                    self.tune_is_calibrating = False
                    self.tune_calibration_status.set("Calibration failed")
                    messagebox.showerror("Auto-Calibrate", f"Calibration failed: {msg.get('error')}")
                    return
        except queue.Empty:
            pass

        self.root.after(100, self._poll_auto_calibration)

    def start_auto_calibration(self):
        if self.tune_is_calibrating:
            return
        if self.tune_image is None:
            messagebox.showwarning("Auto-Calibrate", "Load a tune image first.")
            return

        gt_annotations = self._copy_tune_gt_annotations()
        gt_total = len(gt_annotations['boxes']) + len(gt_annotations['polygons']) + len(gt_annotations['points'])
        if gt_total == 0:
            messagebox.showwarning("Auto-Calibrate", "Draw at least one Ground Truth target first.")
            return

        try:
            target_count = max(1, int(self.tune_target_count.get()))
            self.tune_target_count.set(target_count)
        except Exception:
            messagebox.showwarning("Auto-Calibrate", "Target count must be a valid integer.")
            return

        gray = cv2.cvtColor(self.tune_image, cv2.COLOR_RGB2GRAY)
        init_thresh, init_min_area = self._estimate_initial_thresholds_from_gt(gray, gt_annotations)
        if init_thresh is not None and init_min_area is not None:
            self.tune_skip_slider_callback = True
            self.tune_slider_blob_min_thresh.set(init_thresh)
            self.tune_slider_blob_min_area.set(init_min_area)
            self.tune_skip_slider_callback = False
            self.apply_tune_detection()

        self._cache_current_tune_gt()
        base_settings = self._get_current_tune_settings()
        self.tune_calibration_status.set("Calibrating...")
        self.tune_is_calibrating = True
        self.tune_calibration_queue = queue.Queue()

        snapshot = {
            'frame': self.tune_image.copy(),
            'roi_bbox': tuple(self.tune_roi_bbox),
            'annotations': gt_annotations,
            'target_count': target_count,
            'base_settings': base_settings
        }
        self.tune_calibration_thread = threading.Thread(target=self._auto_calibration_worker, args=(snapshot,), daemon=True)
        self.tune_calibration_thread.start()
        self.root.after(100, self._poll_auto_calibration)
    
    def center_on_detections(self):
        """Auto-zoom map to fit all detection markers."""
        try:
            if not self.marker_manager or not self.map_widget:
                messagebox.showinfo("No Markers", "Marker manager or map not initialized.")
                return
            
            coords = self.marker_manager.get_all_coords()
            if len(coords) == 0:
                messagebox.showinfo("No Markers", "No visible markers to center on.")
                return
            
            # Calculate bounding box
            lats = [c[0] for c in coords]
            lons = [c[1] for c in coords]
            
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            # Add padding
            lat_padding = (max_lat - min_lat) * 0.1 or 0.001
            lon_padding = (max_lon - min_lon) * 0.1 or 0.001
            
            self.map_widget.fit_bounding_box((min_lat - lat_padding, min_lon - lon_padding), 
                                            (max_lat + lat_padding, max_lon + lon_padding))
            self.map_status_label.config(text=f"Centered on {len(coords)} markers")
        except Exception as e:
            print(f"Error centering on detections: {e}")
            self.map_status_label.config(text=f"Centering error: {e}")
    
    def reset_map_view(self):
        """Reset map to initial NSA Annapolis position."""
        try:
            if not self.map_widget:
                self.map_status_label.config(text="Error: Map widget not initialized")
                return
            
            self.map_widget.set_position(self.telemetry_lat, self.telemetry_lon)
            self.map_widget.set_zoom(15)
            self.map_status_label.config(text="View reset to NSA Annapolis")
        except Exception as e:
            print(f"Error resetting map view: {e}")
            self.map_status_label.config(text=f"Reset error: {e}")
    
    def toggle_layer_visibility(self, affiliation):
        """Toggle visibility of markers by affiliation."""
        if not self.marker_manager:
            return
        
        show = self.layer_filters[affiliation].get()
        self.marker_manager.filter_by_affiliation(affiliation, show)
        self.update_marker_count()
        status = "shown" if show else "hidden"
        self.map_status_label.config(text=f"{affiliation} markers {status}")
    
    def update_marker_count(self):
        """Update the marker count display."""
        if not self.marker_manager:
            return
        
        counts = self.marker_manager.get_marker_count_by_affiliation()
        total = sum(counts.values())
        count_text = f"Markers: {total}"
        if counts:
            details = ", ".join([f"{k}:{v}" for k, v in counts.items()])
            count_text += f" ({details})"
        self.marker_count_label.config(text=count_text)
    
    def upload_topo_file(self):
        """Upload and process USGS topographic map file."""
        file_path = filedialog.askopenfilename(
            title="Select USGS Topographic Map",
            filetypes=[
                ("GeoTIFF files", "*.tif *.tiff"),
                ("ZIP files", "*.zip"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.map_status_label.config(text="Processing topo file...")
        self.root.update()
        
        try:
            tif_path = file_path
            
            # If ZIP, extract and find GeoTIFF
            if file_path.lower().endswith('.zip'):
                extract_dir = os.path.join(os.path.dirname(file_path), "topo_extracted")
                os.makedirs(extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find .tif file
                tif_files = [f for f in os.listdir(extract_dir) if f.lower().endswith(('.tif', '.tiff'))]
                if not tif_files:
                    messagebox.showerror("Error", "No GeoTIFF file found in ZIP archive.")
                    return
                tif_path = os.path.join(extract_dir, tif_files[0])
            
            # Try to import rasterio and process
            try:
                import rasterio
                from pyproj import Transformer
                
                with rasterio.open(tif_path) as dataset:
                    # Get bounds
                    bounds = dataset.bounds
                    crs = dataset.crs
                    
                    # Transform to WGS84 if needed
                    if crs is not None and crs.to_string() != 'EPSG:4326':
                        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                        min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
                        max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
                    else:
                        min_lon, min_lat = bounds.left, bounds.bottom
                        max_lon, max_lat = bounds.right, bounds.top
                    
                    # Store overlay data
                    self.topo_overlay = {
                        'path': tif_path,
                        'bounds': ((min_lat, min_lon), (max_lat, max_lon)),
                        'crs': str(crs)
                    }
                    
                    # Zoom to topo area
                    if self.map_widget:
                        self.map_widget.fit_bounding_box((min_lat, min_lon), (max_lat, max_lon))
                    
                    info_text = f"Topo loaded: {os.path.basename(tif_path)}\n"
                    info_text += f"Bounds: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})\n"
                    info_text += f"CRS: {crs}"
                    
                    messagebox.showinfo("Topographic Map Loaded", info_text)
                    self.map_status_label.config(text="Topo overlay loaded")
                    
            except ImportError:
                messagebox.showerror("Error", "rasterio or pyproj not installed. Cannot process GeoTIFF.")
            except Exception as e:
                messagebox.showerror("Error", f"Error processing GeoTIFF: {e}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading topo file: {e}")
            self.map_status_label.config(text="Error loading topo file")

    # --- Mouse Event Handlers for ROI ---
    def on_roi_start(self, event):
        # 1. Check if Manual Mode is ON
        if not self.manual_roi_active.get():
            return
            
        self.rect_start_x = event.x
        self.rect_start_y = event.y
        
        if self.rect_id:
            self.canvas_raw.delete(self.rect_id)
            self.rect_id = None

    def on_roi_drag(self, event):
        # 1. Check if Manual Mode is ON
        if not self.manual_roi_active.get():
            return
        if self.rect_start_x is None:
            return
            
        if self.rect_id: 
            self.canvas_raw.delete(self.rect_id)
            
        # Draw dynamic green box on Raw Feed
        # Changed color to green as requested
        self.rect_id = self.canvas_raw.create_rectangle(
            self.rect_start_x, self.rect_start_y, event.x, event.y, 
            outline='#00ff00', width=2, tag="roi_selector"
        )

    def on_roi_end(self, event):
        # 1. Check if Manual Mode is ON
        if not self.manual_roi_active.get():
            return
            
        if self.rect_start_x is None or self.current_frame_rgb is None: 
            return
        
        x1, y1 = self.rect_start_x, self.rect_start_y
        x2, y2 = event.x, event.y
        
        # Normalize coordinates (ensure x1 < x2)
        x_start, x_end = sorted([x1, x2])
        y_start, y_end = sorted([y1, y2])
        
        # Enforce bounds
        x_start = max(0, x_start); y_start = max(0, y_start)
        x_end = min(self.frame_width, x_end); y_end = min(self.frame_height, y_end)
        
        # Ignore tiny clicks (less than 10px)
        if (x_end - x_start) < 10 or (y_end - y_start) < 10: 
            if self.rect_id: self.canvas_raw.delete(self.rect_id)
            self.rect_start_x = None
            return

        # Store full frame and ROI coordinates
        roi_data = {
            'full_frame': self.current_frame_rgb.copy(),
            'roi_bbox': (x_start, y_start, x_end - x_start, y_end - y_start)  # (x, y, w, h)
        }
        
        # --- FIX: Clear Canvas BEFORE processing ---
        if self.rect_id:
            self.canvas_raw.delete(self.rect_id)
            self.rect_id = None
        self.rect_start_x = None
        
        self.add_detection_to_list(roi_data)

    def add_detection_to_list(self, roi_data):
        # Create thumbnail
        try:
            # Store full frame data with ROI bbox for tuning
            self.captured_images_full.append(roi_data.copy())
            img_index = len(self.captured_images_full) - 1  # Index for this image
            
            # Extract ROI crop for thumbnail display
            full_frame = roi_data['full_frame']
            x, y, w, h = roi_data['roi_bbox']
            roi_img = full_frame[y:y+h, x:x+w]
            
            img_pil = PIL.Image.fromarray(roi_img)
            # Resize to fit sidebar if needed, e.g., max width 120
            base_width = 120
            w_percent = (base_width / float(img_pil.size[0]))
            h_size = int((float(img_pil.size[1]) * float(w_percent)))
            img_pil = img_pil.resize((base_width, h_size), PIL.Image.Resampling.LANCZOS)
            img_tk = PIL.ImageTk.PhotoImage(img_pil)
            self.captured_images.append(img_tk) # Keep ref
            
            # Data Text
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            telemetry_text = (f"Time: {ts}\n"
                              f"Loc: {self.telemetry_lat:.4f}, {self.telemetry_lon:.4f}\n"
                              f"Alt: {self.telemetry_alt}ft | Hdg: {self.telemetry_heading}°")
            
            # Main Row Container
            row_frame = tk.Frame(self.det_frame, bg="#222", pady=5, padx=5, highlightbackground="#444", highlightthickness=1)
            row_frame.pack(fill="x", pady=2, padx=2)
            
            # 1. Thumbnail (Left)
            lbl_img = tk.Label(row_frame, image=img_tk, bg="#222")
            lbl_img.pack(side="left", anchor="n", padx=(0, 5))
            
            # 2. Controls (Right)
            ctrl_frame = tk.Frame(row_frame, bg="#222")
            ctrl_frame.pack(side="left", fill="x", expand=True)
            
            # Telemetry Labels
            lbl_telemetry = tk.Label(ctrl_frame, text=telemetry_text, bg="#222", fg="#00ff00", justify="left", font=("Consolas", 8))
            lbl_telemetry.pack(anchor="w", pady=(0, 5))
            
            # Affiliation
            aff_values = ["Hostile", "Friendly", "Neutral", "Unknown", "Civilian"]
            cb_aff = ttk.Combobox(ctrl_frame, values=aff_values, state="readonly", width=15, font=("Arial", 8))
            cb_aff.set("Affiliation")
            cb_aff.pack(fill="x", pady=1)
            
            # Classification
            class_values = ["Infantry", "Sniper", "Vehicle", "Equipment", "Structure"]
            cb_class = ttk.Combobox(ctrl_frame, values=class_values, state="readonly", width=15, font=("Arial", 8))
            cb_class.set("Classification")
            cb_class.pack(fill="x", pady=1)
            
            # Notes
            ent_notes = tk.Entry(ctrl_frame, bg="#333", fg="white", font=("Arial", 8))
            ent_notes.insert(0, "Notes...")
            ent_notes.pack(fill="x", pady=1)
            
            # "Tune This" Button
            btn_tune = tk.Button(ctrl_frame, text="🔧 Tune This", bg="#f0ad4e", fg="black", 
                                font=("Arial", 8, "bold"), height=1, relief="raised", bd=2,
                                command=lambda idx=img_index: self.load_image_for_tuning(self.captured_images_full[idx]))
            btn_tune.pack(fill="x", pady=(3, 1))
            
            # Add marker to map if marker_manager is available
            if self.marker_manager:
                affiliation = cb_aff.get() if cb_aff.get() != "Affiliation" else "Unknown"
                detection_data = {
                    'timestamp': ts,
                    'lat': self.telemetry_lat,
                    'lon': self.telemetry_lon,
                    'alt': self.telemetry_alt,
                    'heading': self.telemetry_heading,
                    'classification': cb_class.get() if cb_class.get() != "Classification" else "Unknown"
                }
                self.marker_manager.add_marker(affiliation=affiliation, detection_data=detection_data)
                self.update_marker_count()
            
            # Auto-switch to detection tab
            self.notebook.select(self.tab_detections)
            
            # Refresh scrolling
            self.det_frame.update_idletasks()
            self.det_canvas.configure(scrollregion=self.det_canvas.bbox("all"))
            
        except Exception as e:
            print(f"Error adding detection: {e}")

    def clear_detections(self):
        for widget in self.det_frame.winfo_children():
            widget.destroy()
        self.captured_images.clear()
        self.captured_images_full.clear()
        self.det_canvas.configure(scrollregion=self.det_canvas.bbox("all"))

    def update_settings_visibility(self, *args):
        # 1. Hide Everything
        frames = [self.blob_frame, self.edge_frame, self.combined_frame, self.sobel_frame, 
                  self.mog_frame, self.yolo_frame]
        for f in frames:
            f.pack_forget()
        
        method = self.cv_method.get()
        
        # 2. Show Relevant Frames
        if method == 2: # Edge (Canny)
            self.edge_frame.pack(fill="x", pady=5)
        elif method == 3: # Motion (MOG2)
            self.mog_frame.pack(fill="x", pady=5)
        elif method == 4: # Sobel
            self.sobel_frame.pack(fill="x", pady=5)
        elif method == 5: # Combined
            self.combined_frame.pack(fill="x", pady=5)
            self.blob_frame.pack(fill="x", pady=5)
            self.edge_frame.pack(fill="x", pady=5)

    def create_slider(self, parent, label_text, min_val, max_val, default):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill="x", pady=2)
        lbl = tk.Label(frame, text=label_text, bg="#1e1e1e", fg="#aaaaaa", font=("Arial", 8))
        lbl.pack(anchor="w")
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient="horizontal", bg="#1e1e1e", fg="white", 
                          troughcolor="#444", highlightthickness=0, bd=0)
        slider.set(default)
        slider.pack(fill="x")
        return slider

    def reset_trackers(self):
        self.tracker_system.reset()
    
    def toggle_play_pause(self):
        """Toggle between play and pause states."""
        if not hasattr(self, 'cap') or self.cap is None:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.btn_play_pause.config(text="⏸ Pause", bg="#f0ad4e")
        else:
            self.btn_play_pause.config(text="▶ Play", bg="#5cb85c")
    
    def rewind_video(self):
        """Rewind video to the beginning."""
        if not hasattr(self, 'cap') or self.cap is None:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.prev_gray = None
        self.tracker_system.reset()
        self.is_playing = True
        self.btn_play_pause.config(text="⏸ Pause", bg="#f0ad4e")
    
    def stop_video(self):
        """Stop video playback completely."""
        if not hasattr(self, 'cap') or self.cap is None:
            return
        
        self.is_playing = False
        self.stop_event.set()
        
        if self.thread:
            self.thread.join()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear canvases
        self.canvas_raw.delete("all")
        self.canvas_processed.delete("all")
        
        # Disable buttons
        self.btn_play_pause.config(state="disabled", text="▶ Play", bg="#5cb85c")
        self.btn_rewind.config(state="disabled")
        self.btn_stop.config(state="disabled")
        
        self.prev_gray = None
        self.tracker_system.reset()

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.stop_event.set()
            if self.thread:
                self.thread.join()
            self.video_source = file_path
            self.cap = cv2.VideoCapture(self.video_source)
            self.is_playing = True
            self.stop_event.clear()
            self.prev_gray = None 
            self.tracker_system.reset() 
            self.thread = threading.Thread(target=self.video_loop, daemon=True)
            self.thread.start()
            
            # Enable playback control buttons
            self.btn_play_pause.config(state="normal", text="⏸ Pause")
            self.btn_rewind.config(state="normal")
            self.btn_stop.config(state="normal")

    def update_drone_telemetry(self):
        # STATIC VALUES FOR NSA ANNAPOLIS
        self.telemetry_lat = 38.982
        self.telemetry_lon = -76.483
        self.telemetry_alt = 200 # feet
        self.telemetry_heading = 0 # Static North
        
        self.lbl_altitude.config(text=f"Altitude: {self.telemetry_alt} ft")
        self.lbl_gps.config(text=f"GPS: {self.telemetry_lat}, {self.telemetry_lon}")
        self.lbl_grid.config(text=f"Grid Coord: NSA-ANNAP")
        self.lbl_time.config(text=f"Flight Time: -- min")
        self.root.after(1000, self.update_drone_telemetry)

    def video_loop(self):
        while not self.stop_event.is_set() and self.cap.isOpened():
            # Check if video is playing
            if not self.is_playing:
                time.sleep(0.1)  # Sleep while paused to prevent CPU spinning
                continue
            
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.prev_gray = None 
                self.tracker_system.reset()
                continue

            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            if self.pixel_lock_active.get():
                frame = self.stabilize_frame(frame)

            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame_rgb = raw_rgb # Store for ROI capture

            processed_frame = self.process_frame(frame.copy())
            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            self.update_canvas(self.canvas_raw, raw_rgb)
            self.update_canvas(self.canvas_processed, processed_rgb)

            speed = self.playback_speed.get()
            if speed <= 0: speed = 0.1
            delay = 0.033 / speed
            elapsed = time.time() - start_time
            wait = max(0.001, delay - elapsed)
            time.sleep(wait)

    def stabilize_frame(self, curr_frame):
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return curr_frame
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        if prev_pts is None:
            self.prev_gray = curr_gray
            return curr_frame
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts, None)
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        if len(prev_pts) < 10:
            self.prev_gray = curr_gray
            return curr_frame
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        if m is not None:
            dx = m[0, 2]; dy = m[1, 2]; da = np.arctan2(m[1, 0], m[0, 0])
            m_stab = np.zeros((2, 3), np.float32)
            m_stab[0, 0] = np.cos(da); m_stab[0, 1] = -np.sin(da)
            m_stab[1, 0] = np.sin(da); m_stab[1, 1] = np.cos(da)
            m_stab[0, 2] = -dx; m_stab[1, 2] = -dy
            rows, cols = curr_frame.shape[:2]
            curr_frame = cv2.warpAffine(curr_frame, m_stab, (cols, rows))
        self.prev_gray = curr_gray
        return curr_frame

    def process_frame(self, frame):
        # 1. Apply CLAHE if enabled (Pre-processing)
        if self.use_clahe.get():
            frame = self.apply_clahe(frame)

        # 2. Route to Method
        method = self.cv_method.get()
        if method == 2: 
            return self.apply_edge_detection(frame)
        elif method == 3: 
            return self.apply_mog2(frame)
        elif method == 4: 
            return self.apply_sobel(frame)
        elif method == 5: 
            return self.apply_combined(frame)
        else:
            return frame

    def apply_clahe(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = self.slider_clahe_clip.get()
        grid = int(self.slider_clahe_grid.get())
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cv2.putText(final, "CLAHE On", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return final

    def apply_edge_detection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k_size = self.slider_gauss.get()
        if k_size % 2 == 0: k_size += 1 
        blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)
        edges = cv2.Canny(blurred, self.slider_canny_min.get(), self.slider_canny_max.get())
        edges = cv2.dilate(edges, None, iterations=1)
        frame[edges > 0] = [0, 0, 255]
        return frame

    def apply_sobel(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k_size = self.slider_sobel_k.get()
        if k_size % 2 == 0: k_size += 1
        direction = self.sobel_direction.get()
        sobel_x = np.zeros_like(gray); sobel_y = np.zeros_like(gray)
        if direction == "Horizontal" or direction == "Both":
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k_size)
        if direction == "Vertical" or direction == "Both":
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k_size)
        if direction == "Both": grad_mag = cv2.magnitude(sobel_x, sobel_y)
        elif direction == "Horizontal": grad_mag = np.absolute(sobel_x)
        else: grad_mag = np.absolute(sobel_y)
        grad_mag = np.uint8(np.clip(grad_mag, 0, 255))
        mask = grad_mag > 30 
        overlay = frame.copy(); overlay[mask] = [0, 0, 255]
        alpha = 0.1 * self.slider_sobel_weight.get()
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def apply_combined(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        
        # Pre-calc Edges (Only if filtering is enabled)
        filter_edges = self.enable_edge_filter.get()
        edges = None
        if filter_edges:
            k_size = self.slider_gauss.get()
            if k_size % 2 == 0: k_size += 1 
            blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)
            edges = cv2.Canny(blurred, self.slider_canny_min.get(), self.slider_canny_max.get())

        # Blob Detection
        params = cv2.SimpleBlobDetector_Params()
        min_thresh = self.slider_blob_min_thresh.get()
        max_thresh = self.slider_blob_max_thresh.get()
        
        if self.use_otsu.get():
            otsu_val, binary_map = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            min_thresh = otsu_val; max_thresh = 255
            cv2.putText(frame, f"Otsu: {int(otsu_val)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            _, binary_map = cv2.threshold(gray, min_thresh, 255, cv2.THRESH_BINARY)
        
        if min_thresh >= max_thresh: max_thresh = min_thresh + 1
        params.minThreshold = min_thresh; params.maxThreshold = max_thresh
        params.thresholdStep = 10; params.minRepeatability = 1
        params.filterByColor = True; params.blobColor = 255 
        params.filterByArea = True; params.minArea = self.slider_blob_min_area.get(); params.maxArea = 50000 
        params.filterByCircularity = False; params.filterByConvexity = False
        params.filterByInertia = False # Removed Tree filter
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(frame)
        
        detected_rects_scored = []
        min_edge_density = self.slider_edge_density.get() / 100.0

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size * 0.8) 
            x1 = max(0, x - r); y1 = max(0, y - r)
            x2 = min(width, x + r); y2 = min(height, y + r)
            
            roi_bin = binary_map[y1:y2, x1:x2]
            found_rect = None
            if roi_bin.size > 0:
                contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(c)
                    found_rect = (x1 + bx, y1 + by, bw, bh)
            
            if not found_rect:
                r_std = int(kp.size / 2)
                found_rect = (max(0, x - r_std), max(0, y - r_std), r_std*2, r_std*2)

            # EDGE FILTER CHECK
            is_valid = True
            if filter_edges:
                fx, fy, fw, fh = found_rect
                roi_edges = edges[fy:fy+fh, fx:fx+fw]
                if roi_edges.size > 0:
                    density = np.count_nonzero(roi_edges) / roi_edges.size
                    if density < min_edge_density:
                        is_valid = False
                else:
                    is_valid = False
            
            if is_valid:
                fx, fy, fw, fh = found_rect
                roi_gray = gray[fy:fy+fh, fx:fx+fw]
                mean_val = cv2.mean(roi_gray)[0] if roi_gray.size > 0 else 0
                detected_rects_scored.append((found_rect, mean_val))

        detected_rects_scored.sort(key=lambda x: x[1], reverse=True)
        top_rects = [r[0] for r in detected_rects_scored[:5]]
        all_objects = self.tracker_system.update(top_rects)
        
        # Display logic
        tracked_objects_scored = []
        for obj_id, (centroid, box) in all_objects.items():
            tx, ty, tw, th = box
            tx = max(0, tx); ty = max(0, ty)
            roi_gray = gray[ty:ty+th, tx:tx+tw]
            val = cv2.mean(roi_gray)[0] if roi_gray.size > 0 else 0
            tracked_objects_scored.append((obj_id, box, val))
            
        tracked_objects_scored.sort(key=lambda x: x[2], reverse=True)
        
        kept_count = 0
        for i, (obj_id, box, score) in enumerate(tracked_objects_scored):
            if i < 5:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                kept_count += 1
            else:
                self.tracker_system.deregister(obj_id)
                
        cv2.putText(frame, f"Targets: {kept_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def apply_mog2(self, frame):
        history = self.slider_mog_history.get()
        var_thresh = self.slider_mog_var.get()
        self.fgbg.setHistory(history)
        self.fgbg.setVarThreshold(var_thresh)
        fgmask = self.fgbg.apply(frame)
        color_mask = np.zeros_like(frame)
        color_mask[:,:,2] = fgmask 
        result = cv2.addWeighted(frame, 0.7, color_mask, 0.8, 0)
        return result

    def update_canvas(self, canvas, image_data):
        img = PIL.Image.fromarray(image_data)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        canvas.image = imgtk  
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

if __name__ == "__main__":
    root = tk.Tk()
    app = DroneAnalysisApp(root)
    root.mainloop()
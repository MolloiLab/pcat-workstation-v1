"""
PCAT Coronary Artery Contour Editor

Interactive 3D viewer for reviewing and adjusting coronary artery centerlines
and vessel wall radii for PCAT (Pericoronary Adipose Tissue) quantification.

Features:
- 3-plane MPR viewer with vessel centerline overlays
- Drag-adjust centerline points and vessel wall radii
- PCAT volume generation with semi-transparent overlay
- Save adjusted data to files
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from scipy.ndimage import distance_transform_edt

VESSEL_COLORS = {"LAD": "#E8533A", "LCX": "#4A90D9", "RCA": "#2ECC71"}


def launch_coronary_contour_editor(
    volume: np.ndarray,
    spacing_mm: List[float],
    vessel_centerlines: Dict[str, np.ndarray],
    vessel_radii: Dict[str, np.ndarray],
    vessel_voi_masks: Dict[str, np.ndarray],
    output_dir: Path,
    prefix: str,
) -> Dict:
    """
    Launch interactive coronary artery contour editor.
    Returns dict with keys: 'voi_masks', 'centerlines', 'radii'
    (updated by user edits, or original values if no changes)
    """
    editor = CoronaryContourEditor(
        volume, spacing_mm, vessel_centerlines, vessel_radii, 
        vessel_voi_masks, output_dir, prefix
    )
    
    # Run the editor and wait for completion
    editor.run()
    
    # Return updated data
    return {
        'voi_masks': editor.vessel_voi_masks,
        'centerlines': editor.vessel_centerlines,
        'radii': editor.vessel_radii
    }


class CoronaryContourEditor:
    """Interactive coronary artery contour editor with 3-plane MPR viewer."""
    
    def __init__(
        self,
        volume: np.ndarray,
        spacing_mm: List[float],
        vessel_centerlines: Dict[str, np.ndarray],
        vessel_radii: Dict[str, np.ndarray],
        vessel_voi_masks: Dict[str, np.ndarray],
        output_dir: Path,
        prefix: str,
    ):
        self.volume = volume
        self.spacing_mm = spacing_mm
        self.vessel_centerlines = vessel_centerlines
        self.vessel_radii = vessel_radii
        self.vessel_voi_masks = vessel_voi_masks
        self.output_dir = output_dir
        self.prefix = prefix
        
        # Initialize lumen masks dict
        self.vessel_lumen_masks = {}
        
        # PCAT mask
        self.pcat_mask = None
        
        # Current state
        self.active_vessel = "LAD" if "LAD" in vessel_centerlines else list(vessel_centerlines.keys())[0]
        self.z_slice = volume.shape[0] // 2
        self.y_slice = volume.shape[1] // 2
        self.x_slice = volume.shape[2] // 2
        
        # Drag state
        self.dragging = None  # (vessel_name, point_idx)
        self.selected_point_idx = 0
        
        # Window state
        self.fig = None
        self.axes = None
        self.pcat_button = None
        self.status_text = None
        self.info_text = None
        
        # Precompute lumen masks for all vessels
        for vessel_name in self.vessel_centerlines:
            self.vessel_lumen_masks[vessel_name] = np.zeros_like(volume, dtype=bool)
            # Will be filled by _recompute_voi
        
    def run(self):
        """Initialize and run the interactive editor."""
        self._setup_figure()
        self._connect_events()
        self._update_display()
        plt.show()
        
    def _setup_figure(self):
        """Set up the matplotlib figure with 3 image panels and controls."""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Title and instructions
        self.fig.suptitle("=== PCAT Coronary Artery Contour Editor ===", fontsize=14, fontweight='bold')
        self.fig.text(0.5, 0.96, 
                     "Drag centerline points or vessel wall radius circles | 1=LAD 2=LCX 3=RCA | p=Add PCAT | s=Save & Close | q=Quit",
                     ha='center', fontsize=10)
        
        # Create grid layout: 2x3 for main content, plus space for controls
        gs = self.fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.3], height_ratios=[1, 0.05])
        
        # Three image panels
        self.ax_axial = self.fig.add_subplot(gs[0, 0])
        self.ax_coronal = self.fig.add_subplot(gs[0, 1])
        self.ax_sagittal = self.fig.add_subplot(gs[0, 2])
        self.ax_info = self.fig.add_subplot(gs[0, 3])
        
        # PCAT button
        ax_button = self.fig.add_subplot(gs[1, 0])
        self.pcat_button = Button(ax_button, 'Add PCAT', color='yellow')
        self.pcat_button.on_clicked(self._on_pcat_button)
        
        # Status bar
        self.ax_status = self.fig.add_subplot(gs[1, 1:])
        self.status_text = self.ax_status.text(0.5, 0.5, "", ha='center', va='center')
        self.ax_status.axis('off')
        
        # Set titles for image panels
        self.ax_axial.set_title("Axial")
        self.ax_coronal.set_title("Coronal")
        self.ax_sagittal.set_title("Sagittal")
        self.ax_info.set_title("Info")
        
        # Configure axes
        for ax in [self.ax_axial, self.ax_coronal, self.ax_sagittal]:
            ax.axis('off')
        
        self.ax_info.axis('off')
        
    def _connect_events(self):
        """Connect mouse and keyboard events."""
        # Mouse events for dragging
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        
        # Keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
    def _update_display(self):
        """Update all image panels and overlays."""
        # Clear all axes
        for ax in [self.ax_axial, self.ax_coronal, self.ax_sagittal]:
            ax.clear()
            ax.axis('off')
        
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Display slices
        self._display_axial()
        self._display_coronal()
        self._display_sagittal()
        
        # Update info panel
        self._update_info_panel()
        
        # Update status bar
        self._update_status_bar()
        
        # Redraw
        self.fig.canvas.draw_idle()
        
    def _display_axial(self):
        """Display axial view with overlays."""
        # Get current slice
        slice_img = self.volume[self.z_slice].astype(np.float32)
        
        # Display image
        self.ax_axial.imshow(slice_img, cmap='gray', vmin=-300, vmax=300)
        
        # Add PCAT overlay if available
        if self.pcat_mask is not None:
            pcat_slice = self.pcat_mask[self.z_slice]
            yellow = np.zeros((*pcat_slice.shape, 4))
            yellow[pcat_slice] = [1.0, 1.0, 0.0, 0.35]  # Yellow with transparency
            self.ax_axial.imshow(yellow)
        
        # Add vessel overlays
        self._add_vessel_overlays(self.ax_axial, "axial")
        
        self.ax_axial.set_title(f"Axial Z={self.z_slice}")
        
    def _display_coronal(self):
        """Display coronal view with overlays."""
        # Get current slice
        slice_img = self.volume[:, self.y_slice, :].astype(np.float32)
        
        # Display image (flip z axis for better visualization)
        self.ax_coronal.imshow(slice_img.T, cmap='gray', vmin=-300, vmax=300, origin='lower')
        
        # Add PCAT overlay if available
        if self.pcat_mask is not None:
            pcat_slice = self.pcat_mask[:, self.y_slice, :]
            yellow = np.zeros((*pcat_slice.shape, 4))
            yellow[pcat_slice] = [1.0, 1.0, 0.0, 0.35]
            self.ax_coronal.imshow(yellow.T, origin='lower')
        
        # Add vessel overlays
        self._add_vessel_overlays(self.ax_coronal, "coronal")
        
        self.ax_coronal.set_title(f"Coronal Y={self.y_slice}")
        
    def _display_sagittal(self):
        """Display sagittal view with overlays."""
        # Get current slice
        slice_img = self.volume[:, :, self.x_slice].astype(np.float32)
        
        # Display image (flip z axis for better visualization)
        self.ax_sagittal.imshow(slice_img.T, cmap='gray', vmin=-300, vmax=300, origin='lower')
        
        # Add PCAT overlay if available
        if self.pcat_mask is not None:
            pcat_slice = self.pcat_mask[:, :, self.x_slice]
            yellow = np.zeros((*pcat_slice.shape, 4))
            yellow[pcat_slice] = [1.0, 1.0, 0.0, 0.35]
            self.ax_sagittal.imshow(yellow.T, origin='lower')
        
        # Add vessel overlays
        self._add_vessel_overlays(self.ax_sagittal, "sagittal")
        
        self.ax_sagittal.set_title(f"Sagittal X={self.x_slice}")
        
    def _add_vessel_overlays(self, ax, view_type):
        """Add vessel centerline and wall overlays to the specified view."""
        for vessel_name, centerline in self.vessel_centerlines.items():
            if vessel_name not in self.vessel_radii:
                continue
                
            radii = self.vessel_radii[vessel_name]
            color = VESSEL_COLORS.get(vessel_name, 'white')
            
            # Project centerline points and filter by proximity to current slice
            if view_type == "axial":
                # Axial: show points within ±2 slices of current z
                mask = np.abs(centerline[:, 0] - self.z_slice) <= 2
                points = centerline[mask]
                local_radii = radii[mask]
                
                if len(points) > 0:
                    x_coords = points[:, 2]
                    y_coords = points[:, 1]
                    
                    # Plot centerline points
                    ax.scatter(x_coords, y_coords, c=color, s=15, alpha=0.8)
                    
                    # Plot vessel wall circles
                    for i, (x, y, r) in enumerate(zip(x_coords, y_coords, local_radii)):
                        # Convert radius in mm to pixels
                        r_pixels = r / self.spacing_mm[2]
                        circle = plt.Circle((x, y), r_pixels, fill=False, edgecolor=color, 
                                          alpha=0.6, linewidth=1)
                        ax.add_patch(circle)
                    
                    # Highlight selected point
                    if vessel_name == self.active_vessel and self.selected_point_idx < len(points):
                        sel_point = points[self.selected_point_idx]
                        ax.scatter(sel_point[2], sel_point[1], c='yellow', s=50, marker='o', 
                                 edgecolors='black', linewidth=1)
                        
            elif view_type == "coronal":
                # Coronal: show points within ±2 slices of current y
                mask = np.abs(centerline[:, 1] - self.y_slice) <= 2
                points = centerline[mask]
                local_radii = radii[mask]
                
                if len(points) > 0:
                    x_coords = points[:, 2]
                    z_coords = points[:, 0]
                    
                    # Flip z for display
                    flipped_z = self.volume.shape[0] - 1 - z_coords
                    
                    # Plot centerline points
                    ax.scatter(x_coords, flipped_z, c=color, s=15, alpha=0.8)
                    
                    # Plot vessel wall circles
                    for i, (x, z, r) in enumerate(zip(x_coords, z_coords, local_radii)):
                        # Convert radius to pixels
                        r_pixels = r / self.spacing_mm[2]
                        circle = plt.Circle((x, self.volume.shape[0] - 1 - z), r_pixels, 
                                          fill=False, edgecolor=color, alpha=0.6, linewidth=1)
                        ax.add_patch(circle)
                    
                    # Highlight selected point
                    if vessel_name == self.active_vessel and self.selected_point_idx < len(points):
                        sel_point = points[self.selected_point_idx]
                        ax.scatter(sel_point[2], self.volume.shape[0] - 1 - sel_point[0], 
                                 c='yellow', s=50, marker='o', edgecolors='black', linewidth=1)
                        
            elif view_type == "sagittal":
                # Sagittal: show points within ±2 slices of current x
                mask = np.abs(centerline[:, 2] - self.x_slice) <= 2
                points = centerline[mask]
                local_radii = radii[mask]
                
                if len(points) > 0:
                    y_coords = points[:, 1]
                    z_coords = points[:, 0]
                    
                    # Flip z for display
                    flipped_z = self.volume.shape[0] - 1 - z_coords
                    
                    # Plot centerline points
                    ax.scatter(y_coords, flipped_z, c=color, s=15, alpha=0.8)
                    
                    # Plot vessel wall circles
                    for i, (y, z, r) in enumerate(zip(y_coords, z_coords, local_radii)):
                        # Convert radius to pixels
                        r_pixels = r / self.spacing_mm[1]
                        circle = plt.Circle((y, self.volume.shape[0] - 1 - z), r_pixels, 
                                          fill=False, edgecolor=color, alpha=0.6, linewidth=1)
                        ax.add_patch(circle)
                    
                    # Highlight selected point
                    if vessel_name == self.active_vessel and self.selected_point_idx < len(points):
                        sel_point = points[self.selected_point_idx]
                        ax.scatter(sel_point[1], self.volume.shape[0] - 1 - sel_point[0], 
                                 c='yellow', s=50, marker='o', edgecolors='black', linewidth=1)
    
    def _update_info_panel(self):
        """Update the info panel with vessel information."""
        info_text = "Vessel Information:\n\n"
        
        for vessel_name, centerline in self.vessel_centerlines.items():
            if vessel_name not in self.vessel_radii:
                continue
                
            color = VESSEL_COLORS.get(vessel_name, 'white')
            num_points = len(centerline)
            mean_radius = np.mean(self.vessel_radii[vessel_name])
            
            # Highlight active vessel
            active_marker = " ←" if vessel_name == self.active_vessel else ""
            
            info_text += f"{vessel_name}:\n"
            info_text += f"  Color: {color}\n"
            info_text += f"  Points: {num_points}\n"
            info_text += f"  Mean radius: {mean_radius:.2f} mm\n"
            
            # Add PCAT voxel count if available
            if self.pcat_mask is not None and vessel_name in self.vessel_lumen_masks:
                lumen_mask = self.vessel_lumen_masks[vessel_name]
                # Count PCAT voxels for this vessel (approximate)
                # This is not exact since PCAT combines all vessels
                info_text += f"  PCAT: Generated\n"
                
            info_text += active_marker + "\n\n"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top', family='monospace')
        
    def _update_status_bar(self):
        """Update the status bar with current state."""
        status = f"Active: {self.active_vessel} | "
        status += f"Slices: Z={self.z_slice} Y={self.y_slice} X={self.x_slice} | "
        
        if self.active_vessel in self.vessel_centerlines and self.active_vessel in self.vessel_radii:
            if self.selected_point_idx < len(self.vessel_radii[self.active_vessel]):
                current_radius = self.vessel_radii[self.active_vessel][self.selected_point_idx]
                status += f"Point {self.selected_point_idx} radius: {current_radius:.2f} mm"
        
        if self.pcat_mask is not None:
            total_pcat_voxels = np.sum(self.pcat_mask)
            status += f" | PCAT voxels: {total_pcat_voxels}"
            
        if self.dragging:
            vessel, point_idx = self.dragging
            status += f" | DRAGGING {vessel} point {point_idx}"
        
        self.status_text.set_text(status)
        
    def _on_mouse_press(self, event):
        """Handle mouse press events for dragging."""
        if event.inaxes is None:
            return
            
        # Determine which view was clicked
        if event.inaxes == self.ax_axial:
            view_type = "axial"
        elif event.inaxes == self.ax_coronal:
            view_type = "coronal"
        elif event.inaxes == self.ax_sagittal:
            view_type = "sagittal"
        else:
            return
            
        if self.active_vessel not in self.vessel_centerlines:
            return
            
        centerline = self.vessel_centerlines[self.active_vessel]
        
        # Find nearest centerline point within the view
        min_dist = float('inf')
        nearest_idx = None
        
        for i, point in enumerate(centerline):
            if view_type == "axial":
                if abs(point[0] - self.z_slice) > 2:  # Not in current slice range
                    continue
                # Project to 2D
                x, y = point[2], point[1]
            elif view_type == "coronal":
                if abs(point[1] - self.y_slice) > 2:
                    continue
                x, z = point[2], point[0]
                # Flip z for display
                y = self.volume.shape[0] - 1 - z
            elif view_type == "sagittal":
                if abs(point[2] - self.x_slice) > 2:
                    continue
                y, z = point[1], point[0]
                # Flip z for display
                x = self.volume.shape[0] - 1 - z
            else:
                continue
                
            # Calculate 2D distance
            if event.xdata is not None and event.ydata is not None:
                if view_type == "axial":
                    dist = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
                elif view_type == "coronal":
                    dist = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
                elif view_type == "sagittal":
                    dist = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
                    
                if dist < min_dist and dist < 15:  # Within 15 pixels
                    min_dist = dist
                    nearest_idx = i
        
        if nearest_idx is not None:
            self.dragging = (self.active_vessel, nearest_idx)
            self.selected_point_idx = nearest_idx
            
    def _on_mouse_motion(self, event):
        """Handle mouse motion for dragging."""
        if self.dragging is None or event.inaxes is None:
            return
            
        vessel_name, point_idx = self.dragging
        if vessel_name not in self.vessel_centerlines:
            return
            
        # Update centerline point position
        centerline = self.vessel_centerlines[vessel_name]
        
        # Determine which view and update accordingly
        if event.inaxes == self.ax_axial:
            # Axial view - update y and x coordinates
            if event.xdata is not None and event.ydata is not None:
                centerline[point_idx, 1] = int(event.ydata)  # y
                centerline[point_idx, 2] = int(event.xdata)  # x
        elif event.inaxes == self.ax_coronal:
            # Coronal view - update z and x coordinates
            if event.xdata is not None and event.ydata is not None:
                # Convert back from display coordinates
                z = self.volume.shape[0] - 1 - int(event.ydata)
                centerline[point_idx, 0] = z
                centerline[point_idx, 2] = int(event.xdata)
        elif event.inaxes == self.ax_sagittal:
            # Sagittal view - update z and y coordinates
            if event.xdata is not None and event.ydata is not None:
                # Convert back from display coordinates
                z = self.volume.shape[0] - 1 - int(event.ydata)
                centerline[point_idx, 0] = z
                centerline[point_idx, 1] = int(event.xdata)
        
        # Update display
        self._update_display()
        
    def _on_mouse_release(self, event):
        """Handle mouse release to finish dragging."""
        if self.dragging is not None:
            vessel_name, _ = self.dragging
            self._recompute_voi(vessel_name)
            self.dragging = None
            self._update_display()
            
    def _on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'q':
            plt.close('all')
        elif event.key == 's':
            self._save_and_close()
        elif event.key == 'p':
            self._add_pcat()
        elif event.key == '1':
            if 'LAD' in self.vessel_centerlines:
                self.active_vessel = 'LAD'
                self.selected_point_idx = 0
                self._update_display()
        elif event.key == '2':
            if 'LCX' in self.vessel_centerlines:
                self.active_vessel = 'LCX'
                self.selected_point_idx = 0
                self._update_display()
        elif event.key == '3':
            if 'RCA' in self.vessel_centerlines:
                self.active_vessel = 'RCA'
                self.selected_point_idx = 0
                self._update_display()
        elif event.key in ['[', ']']:
            # Radius adjustment
            if self.active_vessel in self.vessel_radii:
                if self.selected_point_idx < len(self.vessel_radii[self.active_vessel]):
                    delta = -0.1 if event.key == '[' else 0.1
                    self.vessel_radii[self.active_vessel][self.selected_point_idx] += delta
                    self._recompute_voi(self.active_vessel)
                    self._update_display()
        elif event.key == 'a':
            # Apply current radius to all points of active vessel
            if self.active_vessel in self.vessel_radii and self.selected_point_idx < len(self.vessel_radii[self.active_vessel]):
                current_radius = self.vessel_radii[self.active_vessel][self.selected_point_idx]
                self.vessel_radii[self.active_vessel][:] = current_radius
                self._recompute_voi(self.active_vessel)
                self._update_display()
        elif event.key == 'up':
            # Navigate slices up
            if self.z_slice < self.volume.shape[0] - 1:
                self.z_slice += 1
                self._update_display()
        elif event.key == 'down':
            # Navigate slices down
            if self.z_slice > 0:
                self.z_slice -= 1
                self._update_display()
        elif event.key == 'left':
            # Navigate point index
            if self.selected_point_idx > 0:
                self.selected_point_idx -= 1
                self._update_display()
        elif event.key == 'right':
            # Navigate point index
            if self.active_vessel in self.vessel_radii and self.selected_point_idx < len(self.vessel_radii[self.active_vessel]) - 1:
                self.selected_point_idx += 1
                self._update_display()
                
    def _on_pcat_button(self, event):
        """Handle PCAT button click."""
        self._add_pcat()
        
    def _add_pcat(self):
        """Generate and display PCAT volume."""
        if self.pcat_mask is not None:
            print("PCAT already generated. Clearing and regenerating...")
            
        # Initialize combined PCAT mask
        self.pcat_mask = np.zeros_like(self.volume, dtype=bool)
        
        # Generate PCAT for each vessel and combine
        for vessel_name in self.vessel_centerlines:
            if vessel_name in self.vessel_lumen_masks:
                vessel_pcat = self._compute_pcat_voi(vessel_name)
                self.pcat_mask |= vessel_pcat
                
        # Subtract all vessel lumen masks from PCAT
        for vessel_name in self.vessel_lumen_masks:
            self.pcat_mask &= ~self.vessel_lumen_masks[vessel_name]
            
        # Print statistics
        total_pcat_voxels = np.sum(self.pcat_mask)
        pcat_volume_ml = total_pcat_voxels * np.prod(self.spacing_mm) / 1000  # Convert to mL
        print(f"PCAT generated: {total_pcat_voxels} voxels ({pcat_volume_ml:.2f} mL)")
        
        # Update display
        self._update_display()
        
    def _recompute_voi(self, vessel_name):
        """
        Rebuild vessel lumen and VOI masks using distance transform.
        
        Args:
            vessel_name: Name of the vessel to recompute
        """
        if vessel_name not in self.vessel_centerlines or vessel_name not in self.vessel_radii:
            return
            
        cl = self.vessel_centerlines[vessel_name]  # (N, 3) [z, y, x]
        radii = self.vessel_radii[vessel_name]
        mean_r = float(np.mean(radii))
        
        # Build tight subvolume around centerline
        margin_vox = np.array([
            int(np.ceil(mean_r * 3.0 / self.spacing_mm[0])) + 3,
            int(np.ceil(mean_r * 3.0 / self.spacing_mm[1])) + 3,
            int(np.ceil(mean_r * 3.0 / self.spacing_mm[2])) + 3,
        ])
        lo = np.maximum(cl.min(axis=0) - margin_vox, 0).astype(int)
        hi = np.minimum(cl.max(axis=0) + margin_vox, np.array(self.volume.shape) - 1).astype(int)
        sub_shape = tuple((hi - lo + 1).tolist())
        
        cl_local = (cl - lo).astype(int)
        cl_mask = np.zeros(sub_shape, dtype=bool)
        for pt in cl_local:
            z, y, x = int(pt[0]), int(pt[1]), int(pt[2])
            if 0 <= z < sub_shape[0] and 0 <= y < sub_shape[1] and 0 <= x < sub_shape[2]:
                cl_mask[z, y, x] = True
        
        dist_mm = distance_transform_edt(~cl_mask, sampling=self.spacing_mm)
        
        # Vessel VOI (shell): inner=mean_r, outer=mean_r*2
        voi_sub = (dist_mm >= mean_r) & (dist_mm <= mean_r * 2.0)
        voi_full = np.zeros(self.volume.shape, dtype=bool)
        voi_full[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1] = voi_sub
        self.vessel_voi_masks[vessel_name] = voi_full
        
        # Vessel lumen
        lumen_sub = dist_mm <= mean_r
        lumen_full = np.zeros(self.volume.shape, dtype=bool)
        lumen_full[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1] = lumen_sub
        self.vessel_lumen_masks[vessel_name] = lumen_full
        
        # If PCAT already computed, recompute PCAT too
        if self.pcat_mask is not None:
            # Clear and regenerate PCAT
            self._add_pcat()
            
    def _compute_pcat_voi(self, vessel_name):
        """
        Compute PCAT VOI for a specific vessel.
        
        Args:
            vessel_name: Name of the vessel
            
        Returns:
            Boolean array for PCAT VOI
        """
        if vessel_name not in self.vessel_centerlines or vessel_name not in self.vessel_radii:
            return np.zeros_like(self.volume, dtype=bool)
            
        cl = self.vessel_centerlines[vessel_name]  # (N, 3) [z, y, x]
        radii = self.vessel_radii[vessel_name]
        mean_r = float(np.mean(radii))
        
        # Build tight subvolume around centerline
        margin_vox = np.array([
            int(np.ceil(mean_r * 3.0 / self.spacing_mm[0])) + 3,
            int(np.ceil(mean_r * 3.0 / self.spacing_mm[1])) + 3,
            int(np.ceil(mean_r * 3.0 / self.spacing_mm[2])) + 3,
        ])
        lo = np.maximum(cl.min(axis=0) - margin_vox, 0).astype(int)
        hi = np.minimum(cl.max(axis=0) + margin_vox, np.array(self.volume.shape) - 1).astype(int)
        sub_shape = tuple((hi - lo + 1).tolist())
        
        cl_local = (cl - lo).astype(int)
        cl_mask = np.zeros(sub_shape, dtype=bool)
        for pt in cl_local:
            z, y, x = int(pt[0]), int(pt[1]), int(pt[2])
            if 0 <= z < sub_shape[0] and 0 <= y < sub_shape[1] and 0 <= x < sub_shape[2]:
                cl_mask[z, y, x] = True
        
        dist_mm = distance_transform_edt(~cl_mask, sampling=self.spacing_mm)
        
        # PCAT: inner=mean_r (vessel wall), outer=mean_r*3.0
        pcat_sub = (dist_mm >= mean_r) & (dist_mm <= mean_r * 3.0)
        pcat_full = np.zeros(self.volume.shape, dtype=bool)
        pcat_full[lo[0]:hi[0]+1, lo[1]:hi[1]+1, lo[2]:hi[2]+1] = pcat_sub
        
        return pcat_full
        
    def _save_and_close(self):
        """Save updated data and close the window."""
        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save updated centerlines and radii to JSON
        contours_data = {}
        for vessel_name in self.vessel_centerlines:
            if vessel_name in self.vessel_radii:
                contours_data[vessel_name] = {
                    "centerline_ijk": self.vessel_centerlines[vessel_name].tolist(),
                    "radii_mm": self.vessel_radii[vessel_name].tolist()
                }
        
        contours_file = self.output_dir / f"{self.prefix}_contours.json"
        with open(contours_file, 'w') as f:
            json.dump(contours_data, f, indent=2)
        
        print(f"Saved contours to {contours_file}")
        
        # Save PCAT mask if available
        if self.pcat_mask is not None:
            pcat_file = self.output_dir / f"{self.prefix}_pcat_mask.npy"
            np.save(pcat_file, self.pcat_mask)
            print(f"Saved PCAT mask to {pcat_file}")
            
        # Write signal file
        signal_file = self.output_dir / f"{self.prefix}_contours.done"
        signal_file.write_text("")
        
        print(f"Saved {self.prefix} contour data")
        
        # Close window
        plt.close('all')


if __name__ == "__main__":
    print("PCAT Coronary Artery Contour Editor")
    print("Use launch_coronary_contour_editor() function to start the editor")
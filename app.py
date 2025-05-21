import subprocess
import sys

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from magiceye_solve.solver import InteractiveSolver
from typing import Tuple

# Set warning mode to ignore or control certain unnecessary warnings
import warnings
warnings.filterwarnings("ignore", message=".*Trying to detect.*share=True.*")

# Store the solver instance globally within the session state if possible,
# or recreate it when needed. Gradio state management can be tricky.
# For simplicity here, we might recreate it or pass it around.

# Placeholder for the solver instance to avoid global state issues if possible
# We will manage this within the Gradio interactions.

def create_autocorrelation_plot(autocorrelation_curve: np.ndarray, current_slider_offset: int, solver_calculated_offset: int, peak_diffs: np.ndarray, peak_indices: np.ndarray = None) -> plt.Figure:
    """
    Generates a matplotlib plot of the autocorrelation curve with vertical lines
    for the current slider offset and the solver's calculated offset.
    
    Args:
        autocorrelation_curve: 1D array of autocorrelation values
        current_slider_offset: Current offset value selected by the user
        solver_calculated_offset: Offset calculated automatically by the solver
        peak_diffs: Differences between consecutive peaks in the autocorrelation curve
        peak_indices: Indices of the peaks in the autocorrelation curve (if available)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if autocorrelation_curve.size > 0:
        # Create x-axis values for proper alignment of the autocorrelation curve
        x_values = np.arange(len(autocorrelation_curve))
        
        # Get the center index of the autocorrelation curve
        center_idx = len(autocorrelation_curve) // 2
        
        # Create shifted x-values so 0 is at the center of the curve
        # This aligns the curve better with the offset values
        shifted_x = x_values - center_idx
        
        # Plot the autocorrelation curve with proper x-axis alignment
        ax.plot(shifted_x, autocorrelation_curve, label="Autocorrelation")
        
        # Mark the current slider offset with a vertical line (adding offset from center)
        ax.axvline(x=current_slider_offset, color='r', linestyle='--', label=f'Current Slider Offset: {current_slider_offset}')

        # Plot solver's calculated offset if it's different from the slider offset
        if solver_calculated_offset != current_slider_offset:
            ax.axvline(x=solver_calculated_offset, color='b', linestyle=':', label=f'Solver\'s Suggested Offset: {solver_calculated_offset}')

        # If we have peak indices, mark them on the plot
        if peak_indices is not None and peak_indices.size > 0:
            # Get y values at the peak indices
            peak_y_values = [autocorrelation_curve[idx] if 0 <= idx < len(autocorrelation_curve) else 0 for idx in peak_indices]
            # Shift peak indices to match the shifted x-axis
            center_idx = len(autocorrelation_curve) // 2
            shifted_peak_indices = peak_indices - center_idx
            ax.scatter(shifted_peak_indices, peak_y_values, color='g', marker='o', label='Detected Peaks')
            
            # Annotate the peak differences if there are consecutive peaks
            if peak_diffs.size > 0:
                for i in range(len(peak_diffs)):
                    if i < len(peak_indices) - 1:  # Make sure we don't go out of bounds
                        # Calculate midpoint using shifted indices
                        mid_x = (shifted_peak_indices[i] + shifted_peak_indices[i+1]) / 2
                        y_pos = min(autocorrelation_curve[peak_indices[i]], autocorrelation_curve[peak_indices[i+1]]) * 0.8
                        diff_val = peak_diffs[i]
                        if diff_val == solver_calculated_offset:  # Highlight the selected difference
                            ax.annotate(f"{diff_val}", 
                                      xy=(mid_x, y_pos), 
                                      xytext=(mid_x, y_pos),
                                      fontsize=9, 
                                      color='red',
                                      bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

        # Add text for max_diff, which is the raw max difference found by the solver
        if peak_diffs.size > 0:
            max_diff_val = np.max(peak_diffs)
            ax.text(0.95, 0.95, f'Raw Max Diff: {max_diff_val}', transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        # Set appropriate x-axis limits to focus on relevant part of the curve
        ax.set_xlim([0, min(700, current_slider_offset * 2, solver_calculated_offset * 2)])
        
        # Add title and labels
        ax.set_title("Autocorrelation Curve")
        ax.set_xlabel("Offset (pixels)")
        ax.set_ylabel("Magnitude")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No autocorrelation data available", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("Autocorrelation Curve")
        ax.set_xlabel("Offset (pixels)")
        ax.set_ylabel("Magnitude")
    plt.close(fig)
    return fig

def solve_and_display(solver_instance: InteractiveSolver, offset_value: int, channel_mode: str) -> Tuple[np.ndarray, plt.Figure]:
    """Solves the image with the given offset and channel mode, returns the result and the updated plot."""
    if solver_instance is None:
        # Handle case where solver hasn't been initialized (e.g., no image uploaded)
        # Return a blank image and an empty plot
        return np.zeros((100, 100), dtype=np.uint8), create_autocorrelation_plot(np.array([]), 0, 0, np.array([]), np.array([])) # Example placeholder

    # Ensure offset is an integer
    offset_value = int(offset_value)

    # Call the solver with the selected channel mode
    solved_image = solver_instance.solve_with_offset(offset_value, channel_mode=channel_mode)

    # Normalize and convert for display if necessary
    if solved_image.size > 0:
        # Check if normalization is needed (solver might return float)
        if solved_image.dtype == float:
             # Basic normalization assuming range isn't drastically outside [0,1] after filters
             solved_image = np.clip(solved_image, np.min(solved_image), np.max(solved_image)) # Clip potential outliers
             min_val, max_val = np.min(solved_image), np.max(solved_image)
             if max_val > min_val:
                  solved_image = (solved_image - min_val) / (max_val - min_val)
             solved_image = (solved_image * 255).astype(np.uint8)
        # Handle cases where the output might be multi-channel concatenated
        # Gradio's Image component expects HWC format.
        # If solver returns M x (N*C), we might need reshaping or averaging.
        # Assuming solver returns grayscale intensity map for now.
        # If it returns M x (N*C), we need to decide how to display it.
        # Let's assume solve_with_offset returns a displayable grayscale image for now.
        # If solve_with_offset returns concatenated color channels, reshaping is needed:
        # if solver_instance.color_image and solved_image.shape[1] == solver_instance.n * solver_instance.c:
        #     final_width_per_channel = solved_image.shape[1] // solver_instance.c
        #     solved_image = solved_image.reshape((solver_instance.m, solver_instance.c, final_width_per_channel))
        #     solved_image = np.transpose(solved_image, (0, 2, 1)) # Reshape to HWC if needed by Gradio

    else:
        # Return a small blank image if solving failed
        solved_image = np.zeros((100, 100), dtype=np.uint8)

    # Handle potential single-channel output from 'average' mode for display
    if solved_image.ndim == 2:
        # Convert grayscale to RGB for consistent display in Gradio Image component
        solved_image = np.stack((solved_image,)*3, axis=-1)

    # Create and return the autocorrelation plot
    autocorrelation_plot_figure = create_autocorrelation_plot(
        solver_instance.autocorrelation_curve,
        offset_value, # Current slider offset
        solver_instance.default_offset, # Solver's calculated offset
        solver_instance.autocorrelation_peak_diffs,
        solver_instance.autocorrelation_peak_indices  # Pass peak indices for visualization
    )

    return solved_image, autocorrelation_plot_figure

def process_image(uploaded_image: np.ndarray):
    """Processes the uploaded image, initializes the solver, and sets up UI."""
    default_channel_mode = "separate" # Default mode for initial solve
    if uploaded_image is None:
        # No image uploaded yet, return default/empty states
        # Return None for the plot as well
        return None, gr.Slider(visible=False), None, gr.Radio(value=default_channel_mode), None # Also return None for the plot

    try:
        # Initialize the solver with the uploaded image
        # The InteractiveSolver handles normalization internally now
        solver = InteractiveSolver(uploaded_image)

        # Determine slider parameters
        min_offset = 1
        max_offset = solver.n - 1 # Max useful offset
        default_offset = solver.default_offset
        # Ensure default is within bounds
        default_offset = max(min_offset, min(default_offset, max_offset))

        # Perform initial solve with the default offset and default channel mode
        initial_solution = solver.solve_with_offset(default_offset, default_channel_mode)

        # Create the initial autocorrelation plot
        initial_plot_figure = create_autocorrelation_plot(
            solver.autocorrelation_curve,
            default_offset, # Current slider offset (initially default)
            solver.default_offset, # Solver's calculated offset
            solver.autocorrelation_peak_diffs,
            solver.autocorrelation_peak_indices  # Pass peak indices for visualization
        )

        # Update UI elements: show solved image, configure and show slider, set channel mode, and show plot
        slider_update = gr.Slider(
            minimum=min_offset,
            maximum=max_offset,
            value=default_offset,
            step=1,
            label=f"Stereogram Offset (Default: {solver.default_offset}, Max: {max_offset})",
            interactive=True,
            visible=True
        )
        # Return the initial solution, updated slider config, solver instance, default channel mode, and initial plot
        return initial_solution, slider_update, solver, gr.Radio(value=default_channel_mode), gr.Plot(value=initial_plot_figure, visible=True)

    except ValueError as e:
        print(f"Error initializing solver: {e}")
        gr.Warning(f"Could not process image. Error: {e}")
        return None, gr.Slider(visible=False), None, gr.Radio(value=default_channel_mode), None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        gr.Error("An unexpected error occurred during image processing.")
        return None, gr.Slider(visible=False), None, gr.Radio(value=default_channel_mode), None


# --- Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Magic Eye Solver")
    gr.Markdown("Upload an autostereogram (Magic Eye image) to reveal the hidden 3D image. Adjust the offset slider to fine-tune the result.")

    # Store the solver instance in Gradio's state
    # Use gr.State() with value=None explicitly
    solver_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Autostereogram")
            gr.Examples(
                examples=[
                    ["examples/1.jpg"],
                    ["examples/2.jpg"],
                    ["examples/3.jpg"],
                    ["examples/4.jpg"],
                    ["examples/5.gif"],
                    ["examples/6.jpg"],
                    ["examples/7.jpg"],
                ],
                inputs=image_input,  # Updated for Gradio 4.x compatibility
                label="Example Images"
            )
            channel_mode_radio = gr.Radio(
                ["separate", "average"],
                label="Color Channel Mode",
                value="separate",
                info="How to handle color channels. 'Separate' processes each independently. 'Average' converts to grayscale first."
            )
            offset_slider = gr.Slider(
                minimum=1, maximum=100, step=1, # Placeholder values
                label="Stereogram Offset",
                interactive=True,
                visible=False # Initially hidden
            )
        with gr.Column(scale=1):
            image_output = gr.Image(label="Solved Image", type="numpy") # Ensure output type matches input for solve_and_display
            autocorrelation_plot = gr.Plot(label="Autocorrelation Curve", visible=False)

    # --- Event Handling ---
    # 1. When a new image is uploaded:
    image_input.change(
        fn=process_image,
        inputs=image_input,
        outputs=[image_output, offset_slider, solver_state, channel_mode_radio, autocorrelation_plot],
        show_progress="full",
        api_name=False  # Disable API exposure for this event
    )

    # 2. When the slider value changes (on release):
    offset_slider.release(
        fn=solve_and_display,
        inputs=[solver_state, offset_slider, channel_mode_radio],
        outputs=[image_output, autocorrelation_plot],
        show_progress="minimal",
        api_name=False  # Disable API exposure for this event
    )

    # 3. When the channel mode changes:
    channel_mode_radio.change(
        fn=solve_and_display,
        inputs=[solver_state, offset_slider, channel_mode_radio],
        outputs=[image_output, autocorrelation_plot],
        show_progress="minimal",
        api_name=False  # Disable API exposure for this event
    )

if __name__ == "__main__":
    # For Hugging Face Spaces compatibility
    demo.queue()  # Enable queuing for better handling of concurrent users
    demo.launch(show_error=True)
    demo.launch(show_error=True)

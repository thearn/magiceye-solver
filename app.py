import subprocess
import sys

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from magiceye_solve.solver import InteractiveSolver
from typing import Tuple

# ignore gradio warnings
import warnings
warnings.filterwarnings("ignore", message=".*Trying to detect.*share=True.*")

# constants
MAX_OFFSET_DISPLAY_LIMIT = 700 # max value for display and slider range for offset

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
            
            # Find which peak corresponds to the selected offset
            selected_peak_idx = None
            for i, peak_idx in enumerate(peak_indices):
                if abs(peak_idx - center_idx) == solver_calculated_offset:
                    selected_peak_idx = i
                    break
            
            # Use different colors for the selected peak vs other peaks
            colors = ['g'] * len(shifted_peak_indices)
            sizes = [30] * len(shifted_peak_indices)
            if selected_peak_idx is not None:
                colors[selected_peak_idx] = 'r'  # Highlight the selected peak in red
                sizes[selected_peak_idx] = 60  # Make the selected peak larger
                
            # Specifically label the peak used for the selected offset
            if selected_peak_idx is not None:
                peak_idx = peak_indices[selected_peak_idx]
                shifted_peak_idx = shifted_peak_indices[selected_peak_idx]
                peak_val = peak_y_values[selected_peak_idx]
                ax.annotate(f"Selected Peak: {solver_calculated_offset}", 
                           xy=(shifted_peak_idx, peak_val), 
                           xytext=(shifted_peak_idx, peak_val*1.1),
                           fontsize=9,
                           color='red',
                           ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='red'))
            
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

        # Set appropriate x-axis limits to focus on relevant part of the curve
        # Use this same limit for the slider's range
        # ensure x_limit is at least a small positive number if offsets are 0 or very small
        relevant_offset_max = max(10, current_slider_offset * 2, solver_calculated_offset * 2) # ensure a minimum sensible view
        x_limit = min(MAX_OFFSET_DISPLAY_LIMIT, relevant_offset_max)
        ax.set_xlim([0, x_limit])
        
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
        # no solver, return blank
        return np.zeros((100, 100), dtype=np.uint8), create_autocorrelation_plot(np.array([]), 0, 0, np.array([]), np.array([]))

    offset_value = int(offset_value)
    solved_image = solver_instance.solve_with_offset(offset_value, channel_mode=channel_mode)

    # normalize for display
    if solved_image.size > 0:
        if solved_image.dtype == float:
            solved_image = np.clip(solved_image, np.min(solved_image), np.max(solved_image))
            min_val, max_val = np.min(solved_image), np.max(solved_image)
            if max_val > min_val:
                solved_image = (solved_image - min_val) / (max_val - min_val)
            solved_image = (solved_image * 255).astype(np.uint8)
    else:
        solved_image = np.zeros((100, 100), dtype=np.uint8)

    if solved_image.ndim == 2:
        solved_image = np.stack((solved_image,) * 3, axis=-1)

    autocorrelation_plot_figure = create_autocorrelation_plot(
        solver_instance.autocorrelation_curve,
        offset_value,
        solver_instance.default_offset,
        solver_instance.autocorrelation_peak_diffs,
        solver_instance.autocorrelation_peak_indices
    )

    return solved_image, autocorrelation_plot_figure

def process_image(uploaded_image: np.ndarray):
    """Processes the uploaded image, initializes the solver, and sets up UI."""
    default_channel_mode = "average" # default mode
    if uploaded_image is None:
        # no image, return empty ui
        return None, gr.Slider(visible=False), None, gr.Radio(value=default_channel_mode), None, gr.Button(visible=False)

    try:
        # init solver
        solver = InteractiveSolver(uploaded_image)

        min_offset = 1
        # ensure max_offset is at least min_offset and considers solver's suggestion
        max_offset = min(MAX_OFFSET_DISPLAY_LIMIT, max(min_offset +1 , solver.default_offset * 2))
        default_offset = solver.default_offset
        default_offset = max(min_offset, min(default_offset, max_offset))

        initial_solution = solver.solve_with_offset(default_offset, default_channel_mode)

        initial_plot_figure = create_autocorrelation_plot(
            solver.autocorrelation_curve,
            default_offset,
            solver.default_offset,
            solver.autocorrelation_peak_diffs,
            solver.autocorrelation_peak_indices
        )

        slider_update = gr.Slider(
            minimum=min_offset,
            maximum=max_offset,
            value=default_offset,
            step=1,
            label=f"Stereogram Offset (Default: {solver.default_offset}, Max: {max_offset})",
            interactive=True,
            visible=True
        )
        return initial_solution, slider_update, solver, gr.Radio(value=default_channel_mode), gr.Plot(value=initial_plot_figure, visible=True), gr.Button(visible=True)

    except ValueError as e:
        print(f"Error initializing solver: {e}")
        return None, gr.Slider(visible=False), None, gr.Radio(value=default_channel_mode), None, gr.Button(visible=False)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, gr.Slider(visible=False), None, gr.Radio(value=default_channel_mode), None, gr.Button(visible=False)

def reset_to_default_offset(solver_instance: InteractiveSolver, channel_mode: str) -> Tuple[np.ndarray, plt.Figure, gr.update]:
    if solver_instance is None:
        # no solver, hide slider
        return np.zeros((100, 100), dtype=np.uint8), create_autocorrelation_plot(np.array([]), 0, 0, np.array([]), np.array([])), gr.update(visible=False)

    default_offset = solver_instance.default_offset
    solved_image, plot_figure = solve_and_display(solver_instance, default_offset, channel_mode)
    updated_slider = gr.update(value=default_offset, label=f"Stereogram Offset (Default: {default_offset})", interactive=True, visible=True)
    return solved_image, plot_figure, updated_slider

# gradio ui
with gr.Blocks() as demo:
    gr.Markdown("# Magic Eye Solver")
    gr.Markdown("Upload an autostereogram (Magic Eye image) to reveal the hidden 3D image. Adjust the offset slider to fine-tune the result.")

    # solver state
    solver_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Autostereogram", height=400)
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
                ["average", "separate"],
                label="Color Channel Mode",
                value="average",
                info="How to handle color channels. 'Separate' processes each independently. 'Average' converts to grayscale first."
            )
        with gr.Column(scale=1):
            image_output = gr.Image(label="Solved Image", type="numpy", height=400)
            autocorrelation_plot = gr.Plot(label="Autocorrelation Curve", visible=False)
            offset_slider = gr.Slider(
                minimum=1, maximum=100, step=1, # placeholder values
                label="Stereogram Offset",
                interactive=True,
                visible=False # hidden by default
            )
            reset_button = gr.Button("Reset Offset", visible=False) # hidden by default

    # event handlers
    image_input.change(
        fn=process_image,
        inputs=image_input,
        outputs=[image_output, offset_slider, solver_state, channel_mode_radio, autocorrelation_plot, reset_button],
        show_progress="full",
        api_name=False
    )

    def solve_and_display_only(solver_instance, offset_value, channel_mode):
        solved_image, plot_figure = solve_and_display(solver_instance, offset_value, channel_mode)
        return solved_image, plot_figure

    offset_slider.release(
        fn=solve_and_display_only,
        inputs=[solver_state, offset_slider, channel_mode_radio],
        outputs=[image_output, autocorrelation_plot],
        show_progress="minimal",
        api_name=False
    )

    channel_mode_radio.change(
        fn=solve_and_display_only,
        inputs=[solver_state, offset_slider, channel_mode_radio],
        outputs=[image_output, autocorrelation_plot],
        show_progress="minimal",
        api_name=False
    )

    reset_button.click(
        fn=reset_to_default_offset,
        inputs=[solver_state, channel_mode_radio],
        outputs=[image_output, autocorrelation_plot, offset_slider],
        show_progress="minimal",
        api_name=False
    )

    gr.Markdown("---")
    gr.Markdown("Find this project on [GitHub](https://github.com/thearn/magiceye-solver)")

if __name__ == "__main__":
    # hf spaces compatibility
    demo.queue()
    demo.launch(show_error=True)

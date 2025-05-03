import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from magiceye_solve.solver import InteractiveSolver

# Store the solver instance globally within the session state if possible,
# or recreate it when needed. Gradio state management can be tricky.
# For simplicity here, we might recreate it or pass it around.

# Placeholder for the solver instance to avoid global state issues if possible
# We will manage this within the Gradio interactions.

def solve_and_display(solver_instance: InteractiveSolver, offset_value: int, channel_mode: str) -> np.ndarray:
    """Solves the image with the given offset and channel mode, returns the result."""
    if solver_instance is None:
        # Handle case where solver hasn't been initialized (e.g., no image uploaded)
        # Return a blank image or placeholder
        return np.zeros((100, 100), dtype=np.uint8) # Example placeholder

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

    return solved_image

def process_image(uploaded_image: np.ndarray):
    """Processes the uploaded image, initializes the solver, and sets up UI."""
    default_channel_mode = "separate" # Default mode for initial solve
    if uploaded_image is None:
        # No image uploaded yet, return default/empty states
        return None, gr.update(visible=False), None, gr.update(value=default_channel_mode) # Image, Slider, Solver State, Channel Mode

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
        initial_solution = solve_and_display(solver, default_offset, default_channel_mode)

        # Update UI elements: show solved image, configure and show slider, set channel mode
        slider_update = gr.update(
            minimum=min_offset,
            maximum=max_offset,
            value=default_offset,
            step=1,
            label=f"Stereogram Offset (Default: {solver.default_offset}, Max: {max_offset})",
            visible=True
        )
        # Return the initial solution, updated slider config, solver instance, and default channel mode
        return initial_solution, slider_update, solver, gr.update(value=default_channel_mode)

    except ValueError as e:
        print(f"Error initializing solver: {e}")
        gr.Warning(f"Could not process image. Error: {e}")
        return None, gr.update(visible=False), None, gr.update(value=default_channel_mode)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        gr.Error("An unexpected error occurred during image processing.")
        return None, gr.update(visible=False), None, gr.update(value=default_channel_mode)


# --- Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Magic Eye Solver")
    gr.Markdown("Upload an autostereogram (Magic Eye image) to reveal the hidden 3D image. Adjust the offset slider to fine-tune the result.")

    # Store the solver instance in Gradio's state
    solver_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Autostereogram")
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

    # --- Event Handling ---
    # --- Event Handling ---
    # 1. When a new image is uploaded:
    image_input.change(
        fn=process_image,
        inputs=[image_input],
        outputs=[image_output, offset_slider, solver_state, channel_mode_radio], # Added channel_mode_radio output
        show_progress="full"
    )

    # 2. When the slider value changes (on release):
    offset_slider.release(
        fn=solve_and_display,
        inputs=[solver_state, offset_slider, channel_mode_radio], # Added channel_mode_radio input
        outputs=[image_output],
        show_progress="minimal"
    )

    # 3. When the channel mode changes:
    channel_mode_radio.change(
        fn=solve_and_display,
        inputs=[solver_state, offset_slider, channel_mode_radio], # Use current slider value
        outputs=[image_output],
        show_progress="minimal"
    )

if __name__ == "__main__":
    demo.launch()

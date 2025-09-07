import numpy as np
import cv2
from PIL import Image
import gradio as gr
import sudoku

def preprocess_image(cell_image: np.ndarray):
    # Resize to 28x28
    image_resized = cv2.resize(cell_image, (28, 28), interpolation=cv2.INTER_AREA)
    # Convert to float32 and scale to [0, 1]
    image_float = image_resized.astype(np.float32) / 255.0
    # Normalize with mean=0.5, std=0.5
    image_normalized = (image_float - 0.5) / 0.5
    # Reshape to [1, 1, 28, 28] for model input
    image_preprocessed = image_normalized.reshape(1, 1, 28, 28)
    return image_preprocessed

def predict(cell_image: np.ndarray, onnx_model_path: str) -> int:
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Preprocess the cell image
    image_input = preprocess_image(cell_image)
    # Load the ONNX model
    net = cv2.dnn.readNetFromONNX(onnx_model_path)
    # Set input for the network
    net.setInput(image_input)
    # Run inference
    output = net.forward()
    # Get predicted digit (index of max logit)
    predicted_class = np.argmax(output, axis=1)[0]
    predicted_digit = classes[predicted_class]
    return predicted_digit

def recognize_digits_onnx(grid_image: np.ndarray, onnx_model_path: str) -> np.ndarray:
    arr = np.zeros((9,9), dtype=np.uint8)
    cells = sudoku.extract_cells(grid_image)
    for row in range(9):
        for col in range(9):
            cell = cells[row, col]
            if not sudoku.is_empty_cell(cell):
                predicted_digit = predict(cell, onnx_model_path)
                arr[row, col] = predicted_digit
    return arr

def sudoku_solver(sudoku_image: str):
    sudoku_grid = sudoku.extract_grid(sudoku_image, size=9*50)
    if sudoku_grid is not None:
        puzzle = recognize_digits_onnx(sudoku_grid, 'digits.onnx')
        puzzle_copy = puzzle.copy()
        if sudoku.solve(puzzle):
            solution = sudoku.get_solution(puzzle_copy, puzzle)
            sudoku.draw_solution(sudoku_grid, solution, (0,0,255))
            sudoku_grid = cv2.cvtColor(sudoku_grid, cv2.COLOR_BGR2RGB)
            sudoku_grid = Image.fromarray(sudoku_grid)
            return sudoku_grid, 'Puzzle Solved!'
        else:
            sorry = Image.open('images/sorry.jpg')
            return sorry, "Puzzle is unsolvable!"
    else:
        sorry = Image.open('images/sorry.jpg')
        return sorry, "Can't extract sudoku grid!"

examples = [
    ['images/example_01.jpg'],
    ['images/example_02.jpg'],
    ['images/example_03.jpg'],
    ['images/example_04.jpg'],
    ['images/example_05.jpg'],
    ['images/example_06.jpg']
]

app = gr.Interface(
    fn=sudoku_solver,
    inputs=['image'],
    outputs=['image', 'text'],
    title='\U0001F439 Sudoku Solver \U0001F439',
    examples=examples
)

app.launch()

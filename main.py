from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import sudoku
import io
import base64

app = FastAPI()

# Add CORS middleware to allow requests from different origins
# In production, you should restrict this to your frontend's domain for security
# e.g., origins=["https://your-frontend-domain.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

def create_sorry_image(text):
    sorry_img = np.zeros((450, 450, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sorry_img, text, (50, 225), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(sorry_img, cv2.COLOR_BGR2RGB))


def sudoku_solver(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    sudoku_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
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
            sorry = create_sorry_image("Puzzle is unsolvable!")
            return sorry, "Puzzle is unsolvable!"
    else:
        sorry = create_sorry_image("Can't extract sudoku grid!")
        return sorry, "Can't extract sudoku grid!"

@app.post("/solve/")
async def solve_sudoku(file: UploadFile = File(...)):
    image_bytes = await file.read()

    solved_image, message = sudoku_solver(image_bytes)

    buffered = io.BytesIO()
    solved_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return JSONResponse(content={"image": img_str, "message": message})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
